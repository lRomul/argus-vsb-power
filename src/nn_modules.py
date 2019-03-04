import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True):
        super().__init__()

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)


class SimpleLSTM(nn.Module):
    def __init__(self, seq_len, input_size, p_dropout=0.2, base_size=64):
        super().__init__()

        self.lstm1 = nn.LSTM(input_size, base_size*2, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(base_size*4, base_size, bidirectional=True, batch_first=True)

        self.attention = Attention(base_size*2, seq_len)

        self.fc1 = nn.Linear(base_size*2, base_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p_dropout)
        self.fc2 = nn.Linear(base_size, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        x = self.attention(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class BasicConv1d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size,
                 stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_planes, out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            dilation=dilation)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1dFeatureExtractor(nn.Module):
    def __init__(self, input_size=1, base_size=64, p_dropout=0.1):
        super().__init__()
        self.channels = input_size
        self.s = base_size
        self.dropout = p_dropout
        self.input_conv = BasicConv1d(input_size, base_size//4, 1)
        self.conv_1 = BasicConv1d(base_size//4, base_size*1, 4, stride=2)
        self.conv_2 = BasicConv1d(base_size*1, base_size*1, 4, stride=2)
        self.conv_3 = BasicConv1d(base_size*1, base_size*2, 4, stride=2)
        self.conv_4 = BasicConv1d(base_size*2, base_size*2, 4, stride=2)
        self.conv_5 = BasicConv1d(base_size*2, base_size*4, 4, stride=2)
        self.conv_6 = BasicConv1d(base_size*4, base_size*4, 4, stride=2)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x):
        x = self.input_conv(x)

        x = self.conv_1(x)
        x = self.pool(x)
        x = self.conv_2(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = self.conv_3(x)
        x = self.pool(x)
        x = self.conv_4(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = self.conv_5(x)
        x = self.pool(x)
        x = self.conv_6(x)
        x = self.pool(x)
        x = self.dropout(x)

        return x


class Conv1dAvgPool(nn.Module):
    def __init__(self, input_size, base_size=64,
                 conv_dropout=0.1, fc_dropout=0.1):
        super().__init__()

        self.conv = Conv1dFeatureExtractor(input_size, base_size//4, conv_dropout)

        self.fc1 = nn.Linear(base_size, base_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(fc_dropout)
        self.fc2 = nn.Linear(base_size, 3)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class Conv1dLSTMAtt(nn.Module):
    def __init__(self, input_size, base_size=64, seq_len=194,
                 conv_dropout=0.1, fc_dropout=0.1):
        super().__init__()

        self.conv = Conv1dFeatureExtractor(input_size, base_size//4, conv_dropout)

        self.lstm = nn.LSTM(base_size, base_size, num_layers=2,
                             bidirectional=True, batch_first=True)
        self.attention = Attention(base_size*2, seq_len)

        self.fc1 = nn.Linear(base_size*2, base_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(fc_dropout)
        self.fc2 = nn.Linear(base_size, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 1)

        x, _ = self.lstm(x)
        x = self.attention(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class Conv1dAttPNFeatureExtractor(nn.Module):
    def __init__(self, input_size=1, base_size=64, p_dropout=0.1):
        super().__init__()
        self.channels = input_size
        self.s = base_size
        self.dropout = p_dropout
        self.input_conv = BasicConv1d(input_size, base_size//4, 1)
        self.conv_1 = BasicConv1d(base_size//4, base_size*1, 4, stride=2)
        self.conv_2 = BasicConv1d(base_size*1, base_size*1, 4, stride=2)
        self.conv_3 = BasicConv1d(base_size*1, base_size*2, 4, stride=2)
        self.conv_4 = BasicConv1d(base_size*2, base_size*2, 4, stride=2)
        self.conv_5 = BasicConv1d(base_size*2, base_size*4, 4, stride=2)
        self.conv_6 = BasicConv1d(base_size*4, base_size*4, 4, stride=2)

        self.lstm2 = nn.LSTM(base_size * 2, base_size * 2, num_layers=2,
                             bidirectional=True, batch_first=True)
        self.att2 = Attention(base_size * 4, 3124)

        self.lstm3 = nn.LSTM(base_size * 4, base_size * 4, num_layers=2,
                             bidirectional=True, batch_first=True)
        self.att3 = Attention(base_size * 8, 194)

        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x):
        x = self.input_conv(x)

        x = self.conv_1(x)
        x = self.pool(x)
        x = self.conv_2(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = self.conv_3(x)
        x = self.pool(x)
        x = self.conv_4(x)
        x = self.pool(x)
        x = self.dropout(x)

        y = x.permute(0, 2, 1)
        y, _ = self.lstm2(y)
        y = self.att2(y)

        x = self.conv_5(x)
        x = self.pool(x)
        x = self.conv_6(x)
        x = self.pool(x)
        x = self.dropout(x)

        z = x.permute(0, 2, 1)
        z, _ = self.lstm3(z)
        z = self.att3(z)

        x = torch.cat([y, z], dim=1)

        return x


class Conv1dAttPN(nn.Module):
    def __init__(self, input_size, base_size=64,
                 conv_dropout=0.1, fc_dropout=0.1):
        super().__init__()

        self.pn = Conv1dAttPNFeatureExtractor(input_size, base_size//4, conv_dropout)

        fc_size = (base_size // 16) * 48

        self.fc1 = nn.Linear(fc_size, fc_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(fc_dropout)
        self.fc2 = nn.Linear(fc_size // 2, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pn(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class Conv1dAttPNNPFeatureExtractor(nn.Module):
    def __init__(self, input_size=1, base_size=64, p_dropout=0.1):
        super().__init__()
        self.channels = input_size
        self.s = base_size
        self.dropout = p_dropout
        self.input_conv = BasicConv1d(input_size, base_size//4, 1)

        self.conv_1 = BasicConv1d(base_size//4, base_size*1, 4, stride=2)
        self.conv_11 = BasicConv1d(base_size*1, base_size * 1, 4, stride=2)
        self.conv_2 = BasicConv1d(base_size*1, base_size*1, 4, stride=2)
        self.conv_22 = BasicConv1d(base_size*1, base_size*1, 4, stride=2)

        self.conv_3 = BasicConv1d(base_size*1, base_size*2, 4, stride=2)
        self.conv_33 = BasicConv1d(base_size*2, base_size * 2, 4, stride=2)
        self.conv_4 = BasicConv1d(base_size*2, base_size*2, 4, stride=2)
        self.conv_44 = BasicConv1d(base_size*2, base_size*2, 4, stride=2)

        self.conv_5 = BasicConv1d(base_size*2, base_size*4, 4, stride=2)
        self.conv_55 = BasicConv1d(base_size*4, base_size * 4, 4, stride=2)
        self.conv_6 = BasicConv1d(base_size*4, base_size*4, 4, stride=2)
        self.conv_66 = BasicConv1d(base_size*4, base_size*4, 4, stride=2)

        self.lstm2 = nn.LSTM(base_size * 2, base_size * 2, num_layers=2,
                             bidirectional=True, batch_first=True)
        self.att2 = Attention(base_size * 4, 3123)

        self.lstm3 = nn.LSTM(base_size * 4, base_size * 4, num_layers=2,
                             bidirectional=True, batch_first=True)
        self.att3 = Attention(base_size * 8, 193)

        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x):
        x = self.input_conv(x)

        x = self.conv_1(x)
        x = self.conv_11(x)
        x = self.conv_2(x)
        x = self.conv_22(x)
        x = self.dropout(x)

        x = self.conv_3(x)
        x = self.conv_33(x)
        x = self.conv_4(x)
        x = self.conv_44(x)
        x = self.dropout(x)

        y = x.permute(0, 2, 1)
        y, _ = self.lstm2(y)
        y = self.att2(y)

        x = self.conv_5(x)
        x = self.conv_55(x)
        x = self.conv_6(x)
        x = self.conv_66(x)
        x = self.dropout(x)

        z = x.permute(0, 2, 1)
        z, _ = self.lstm3(z)
        z = self.att3(z)

        x = torch.cat([y, z], dim=1)

        return x

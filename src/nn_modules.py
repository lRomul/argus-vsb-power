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
    def __init__(self, input_size=1, base_size=64, p_dropout=0.25):
        super().__init__()
        self.channels = input_size
        self.s = base_size
        self.dropout = p_dropout
        self.input_conv = BasicConv1d(input_size, base_size, 1)
        self.conv_1 = BasicConv1d(base_size*1, base_size*1, 2, stride=2)
        self.conv_2 = BasicConv1d(base_size*1, base_size*1, 2, stride=2)
        self.conv_3 = BasicConv1d(base_size*1, base_size*2, 2, stride=2)
        self.conv_4 = BasicConv1d(base_size*2, base_size*2, 2, stride=2)
        self.conv_5 = BasicConv1d(base_size*2, base_size*4, 2, stride=2)
        self.conv_6 = BasicConv1d(base_size*4, base_size*4, 2, stride=2)
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


class Conv1dAttention(nn.Module):
    def __init__(self, seq_len, input_size, p_dropout=0.2, base_size=64):
        super().__init__()

        self.conv = Conv1dFeatureExtractor(input_size, base_size//4, p_dropout)
        self.attention = Attention(base_size, seq_len)

        self.fc1 = nn.Linear(base_size, base_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p_dropout)
        self.fc2 = nn.Linear(base_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 1)

        x = self.attention(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class SimpleLSTM(nn.Module):
    def __init__(self, seq_len, input_size, p_dropout=0.2, base_size=64):
        super().__init__()

        self.lstm1 = nn.LSTM(input_size, base_size*2, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(base_size*4, base_size, bidirectional=True, batch_first=True)

        self.attention = Attention(base_size*2, seq_len)

        self.fc1 = nn.Linear(base_size*2, base_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p_dropout)
        self.fc2 = nn.Linear(base_size, 1)
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

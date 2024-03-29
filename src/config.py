from os.path import join

DATA_DIR = '/workdir/data/'
METADATA_TEST_PATH = join(DATA_DIR, 'metadata_test.csv')
METADATA_TRAIN_PATH = join(DATA_DIR, 'metadata_train.csv')
TEST_PARQUET_PATH = join(DATA_DIR, 'test.parquet')
TRAIN_PARQUET_PATH = join(DATA_DIR, 'train.parquet')
SAMPLE_SUBMISSION = join(DATA_DIR, 'sample_submission.csv')
TRAIN_FOLDS_PATH = join(DATA_DIR, 'train_folds.csv')
EXPERIMENTS_DIR = join(DATA_DIR, 'experiments')

N_FOLDS = 5
FOLDS = list(range(N_FOLDS))

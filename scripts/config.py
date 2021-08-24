# config.py
# define all the configuration here

TRAIN_BATCH_SIZE = 128
VAL_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32

# 
MAX_LENGTH = 32


# max epoch
EPOCHS = 300


# loss
ALPHA = 3 # 1-5

# learning rate
LEARNING_RATE = 1e-3


# the parameter use to select model
SELECTED = 'loss'  # F1score/loss

# dataset file path
TRAIN_DATASET_FNAME = '../inputs/datasets/train_dataset.csv'
TEST_DATASET_FNAME = '../inputs/datasets/valid_dataset2.csv'

# embedding
EMBEDDING_FNAME = '../inputs/embeddings/glove.6B.50d.txt'
#EMBEDDING_FNAME = '../inputs/embeddings/type_embedding.txt' # 太惨了0.35

# type of network
NETWORK = 'TextCNN' # TextCNN textRNN textRNN_Att DPCNN fastText TextRCNN Transformer





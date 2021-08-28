# config.py

import transformers
import os 
# from transformers import BertTokenizer

# this is the maximum number of tokens in the sentence
MAX_LEN = 30

# this is the maximum number of tokens in the sentence
TRAIN_BATCH_SIZE = 32  # 1
VALIDATION_BATCH_SIZE = 4

# type of network
RANDOM_SEED = 1234
NETWORK = 'BERT_NN'

# Learning rate
LEARNING_RATE = 5e-5

# maximum train epochs
EPOCHS = 10

# 
NUM_CLASS = 2

# 
DROPOUT_RATE = 0.3

# 
TRAIN_SIZE = 0.8

# the parameter use to select model
SELECTED = 'loss'  # F1_score/loss

# define path to BERT model files
BERT_PATH = 'bert-base-uncased'

# file to save the model
MODEL_PATH = "pytorch_model.bin"


# train dataset file
DATASET_PATH = '../inputs/datasets/'
TRAIN_DATASET_FNAME = 'train_dataset.csv'
TEST_DATASET_FNAME = 'valid_dataset2.csv'
TRAIN_DATASET_PATH = os.path.join(DATASET_PATH, TRAIN_DATASET_FNAME)
TEST_DATASET_PATH = os.path.join(DATASET_PATH, TEST_DATASET_FNAME)

# Bert
TOKENIZER = transformers.BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)





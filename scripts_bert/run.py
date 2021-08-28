# train.py

import io
import torch
# import torch.utils.data as Data
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.autograd.grad_mode import no_grad 
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import os
import datetime

from sklearn import metrics
from sklearn import model_selection
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import torch.optim as optim
from sklearn import metrics
from sklearn.utils import shuffle

import dataset
import config 
import engine
from importlib import import_module
import metrics_func

def train():
    """
    This function trains the model
    read the training file and fill NaN values with "none"
    can also choose to drop NaN values in this specific dataset
    """
    
    # split dataset
    data_df = pd.read_csv(config.TRAIN_DATASET_PATH)
    # shuffle dataset
    train_df = shuffle(data_df)
#     # 8:2
#     train_df = data_df.sample(frac=config.TRAIN_SIZE,random_state=0,axis=0)
#     validation_df = data_df[~data_df.index.isin(train_df.index)]
  
    test_df = pd.read_csv(config.TEST_DATASET_PATH)
    print('train size: ',len(train_df))
#     print('validation size: ',len(validation_df))
    print('test size: ',len(test_df))

    # initialize BERT Dataset from dataset.py
    train_dataset = dataset.BertDataset(
        entity1=train_df.entity1.values,
        text=train_df.text.values,
        label=train_df.label.values
    )

    # create training dataloader
    train_data_loader = DataLoader(
        train_dataset, # the training samples
        sampler = RandomSampler(train_dataset), # Select batches randomly
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=4
    )

#     # initialize BERTDataset from dataset.py for validaton dataset
#     validation_dataset = dataset.BertDataset(
#         entity1=train_df.entity1.values,
#         text=validation_df.text.values,
#         label=validation_df.label.values
#     )

#     # create validation data loader
#     validation_data_loader = DataLoader(
#         validation_dataset, # the validation samples.
#         batch_size= config.VALIDATION_BATCH_SIZE, # Pull out batches sequentially.
#         num_workers=1
#     )
    
    # test
    test_dataset = dataset.BertDataset(
        entity1=train_df.entity1.values,
        text = test_df.text.values,
        label = test_df.label.values
    )
    test_data_loader = DataLoader(
        test_dataset, # the validation samples.
        batch_size= config.VALIDATION_BATCH_SIZE, # Pull out batches sequentially.
        num_workers=1
    )

    # initialize the cuda device
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    print("device:",device)
    # load model and send it to the device
    torch.manual_seed(config.RANDOM_SEED)
    print(config.NETWORK)
    x = import_module('networks.'+config.NETWORK)
    model = x.Model()
    model.to(device)

    # create parameters we want to optimize
    # we generally do not use any decay for bias and weight layers
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias","LayerNorm.bias","LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [p for n,p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay":0.001,
        },
        {
            "params": [p for n,p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay":0.0,
        },
    ]

    # calculate the number of training steps
    # this is used by scheduler
    num_train_steps = int(len(train_df) / config.TRAIN_BATCH_SIZE * config.EPOCHS)

    # AdamW optimizer, the most widely used optimzer for transformer based networks
    optimizer = AdamW(optimizer_parameters, lr=config.LEARNING_RATE)
#     optimizer = optim.Adam(model.parameters(),lr=config.LEARNING_RATE)
    

    # fetch a scheduler
    # you can also try using reduce lr on plateau
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )

    # if you have multiple GPUs, model model to DataParallel to use multiple GPUs
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    params_list = []

    # start training the epochs
    for epoch in range(config.EPOCHS):
        epoch_start_time = time.time()
        # train one epoch
        train_outputs, train_labels, train_loss = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        # test
        test_outputs, test_labels = engine.eval_fn(test_data_loader, model, device)
        
        ###----Train--------
        train_outputs = torch.Tensor(train_outputs)
        _, train_predicted = torch.max(train_outputs, dim=1)
        train_parameters_dict = metrics_func.performance_evaluation_func(train_predicted,train_labels,epoch=str(epoch),loss=train_loss)
        # save train paremeters
        params_list.append(train_parameters_dict)
        
        train_f1 = train_parameters_dict['f1_score_macro']
        train_prec = train_parameters_dict['precision_macro']
        train_recall = train_parameters_dict['precision_macro']
        print(f" Train Epoch: {epoch}, F1 = {train_f1},precision = {train_prec},recall = {train_recall}")

        ###-------Test-----------------------
        test_outputs = torch.Tensor(test_outputs)
        _, test_predicted = torch.max(test_outputs, dim=1)    
        # calculate evaluation paremeters
        test_parameters_dict = metrics_func.performance_evaluation_func(test_predicted, test_labels, epoch=str(epoch),flag='test')
        # save evaluation paremeters
        params_list.append(test_parameters_dict)
           
        test_f1 = test_parameters_dict['f1_score_macro']
        test_prec = test_parameters_dict['precision_macro']
        test_recall = test_parameters_dict['recall_macro']
        print(f"test Epoch: {epoch},F1 = {test_f1},precision = {test_prec}, recall = {test_recall}")
        print('\n')
        
        #save_model_func(model, epoch, path='outputs')
    
    metrics_func.save_parameters_txt(params_list)

if __name__ == "__main__":
  
    train()



    
    

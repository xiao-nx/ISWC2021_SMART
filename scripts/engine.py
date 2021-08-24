# engine.py

import torch
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F
import config
import numpy as np

def train_fn(data_loader, model, optimizer,device):
    """
    training function which trains for one epoch
    parameters:
    data_loader: torch dataloader object
    model: torch model,bert in our case
    optimizer:torch optimizer, e.g. adam,sgd, etc.
    device: "cuda" or "cpu"
    scheduler: learning rate scheduler
    """
    
    model.train()
    
    # Adjust learning rate
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
    
    final_labels = []
    final_outputs = []
    total_loss = []
    
    # loop over all batches
    for data in data_loader:
        entity1, text, label = data.entity1, data.text, data.label
        entity1 = (entity1.unsqueeze(0)).expand(text.shape)
        
        train_data = {'entity1':entity1,'text':text}
        
        # zero-grad the optimizer
        optimizer.zero_grad()
            
        # pass through the model
        logits = model(train_data)
        
        # cross entropy loss for classifier
        #loss = nn.BCEWithLogitsLoss()(logits, label)
        loss = nn.CrossEntropyLoss()(logits, label)

        # backward step the loss
        loss.backward()
        
        # step optimizer
        optimizer.step()
        
        # save training parameters
        final_labels.extend(label.cpu().detach().numpy().tolist())
        final_outputs.extend(logits.cpu().detach().numpy().tolist())
        total_loss.append(loss.cpu().detach().numpy().tolist())
        
    train_loss = np.average(total_loss)
    
    # Adjust learning rate
    scheduler.step()
        
    return final_outputs, final_labels, train_loss 
    
def evaluate_fn(data_loader, model, device,loss_flag=False):
    # initialize empty lists to store predictions and labels
    final_predictions = []
    final_labels = []
    total_loss =[]

    # put the model in eval mode
    model.eval()
    
    # disable gradient calculation
    with torch.no_grad():
        
        for data in data_loader:
            # fetch text and label from the dict
            entity1, text, label = data.entity1, data.text, data.label
            entity1 = (entity1.unsqueeze(0)).expand(text.shape)
            
            val_data = {'entity1':entity1,'text':text}
            
            # make predictions
            logits = model(val_data)
            
            loss = nn.CrossEntropyLoss()(logits, label)
            total_loss.append(loss.cpu().detach().numpy().tolist())

            # move predictions and labels to list, move predictiona and labels to cpu.
            logits = logits.cpu().numpy().tolist()
            labels = data.label.cpu().numpy().tolist()
            final_predictions.extend(logits)
            final_labels.extend(labels)
    
    valid_loss = np.average(total_loss)

    # return final predictions and labels
    return final_predictions,final_labels, valid_loss

    






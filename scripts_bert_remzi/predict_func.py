import io
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.autograd.grad_mode import no_grad 
import numpy as np
import pandas as pd
import time
import os
import datetime
from sklearn import metrics
from torch.utils.data import TensorDataset

from transformers import BertForSequenceClassification

import dataset
import config 


def predict_func():
    
    test_df = pd.read_csv(config.TEST_DATASET_PATH)
    
    # Create sentence and label lists
    true_labels = []

    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []

    # For every sentence...
    for i,row in test_df.iterrows():
        #sent =id2question[row['id']]+'[SEP]'+row['specific_type']
        sent = row['text']
        #if row['label'] not in label_to_id: continue

        #true_labels.append(int(label_to_id[row['specific_type']]))
        true_labels.append(row['label'])
        tokenizer = config.TOKENIZER
        encoded_dict = tokenizer.encode_plus(
                                sent,                      # Sentence to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                max_length = 75,           # Pad & truncate all sentences.
                                pad_to_max_length = True,
                                return_attention_mask = True,   # Construct attn. masks.
                                return_tensors = 'pt',     # Return pytorch tensors.
                           )

        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])
        #true_labels.append(int(row['class']))

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    # labels = torch.tensor(labels)

    # Set the batch size.  
    batch_size = 32  

    # Create the DataLoader.
    # prediction_data = TensorDataset(input_ids, attention_masks, labels)
    prediction_data = TensorDataset(input_ids, attention_masks)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)    

    print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))
    
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    
    model = BertForSequenceClassification.from_pretrained(
               'bert-base-uncased', 
               num_labels = 2, # The number of output labels. 2 for binary classification.  
               output_attentions = False,
               output_hidden_states = False)
    
    model.load_state_dict(torch.load(config.model_dir))
    model = model.to(device)
    # Put model in evaluation mode
    model.eval()

    # Tracking variables 
    predictions = []
#     print('length: ',len(prediction_dataloader))

    # Predict 
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        
        # Unpack the inputs from our dataloader
        # b_input_ids, b_input_mask, b_labels = batch
        b_input_ids, b_input_mask = batch
        
        # Telling the model not to compute or store gradients, saving memory and speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(
                            b_input_ids, 
                            token_type_ids=None,
                            attention_mask=b_input_mask
                           )
        logits = outputs[0]
        
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        
        # Store predictions and true labels
        predictions.append(logits)
        
    print('    DONE.')    
    
    # # Combine the results across all batches. 
    flat_predictions = np.concatenate(predictions, axis=0)

    # # For each sample, pick the label (0 or 1) with the higher score.
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten() 
    acc = metrics.accuracy_score(true_labels, flat_predictions)
    print('predicte acc: ',acc)


if __name__ == "__main__":
    #train()
    predict_func()
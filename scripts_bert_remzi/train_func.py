# train.py

import io
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.autograd.grad_mode import no_grad 
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import os
import datetime
import random
from sklearn import metrics
from torch.utils.data import TensorDataset

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification

import dataset
import config 

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
    

def train():
    """
    This function trains the model
    read the training file and fill NaN values with "none"
    can also choose to drop NaN values in this specific dataset
    """
    
    # split dataset
    train_df = pd.read_csv(config.TRAIN_DATASET_PATH)
    test_df = pd.read_csv(config.TEST_DATASET_PATH)
    print('train size: ',len(train_df))
    print('test size: ',len(test_df))

    # initialize BERT Dataset from dataset.py
    train_dataset = dataset.BertDataset(    
        text=train_df.text.values,
        label=train_df.label.values
    )

    # create training dataloader
    train_dataloader = DataLoader(
        train_dataset, # the training samples
        sampler = RandomSampler(train_dataset), # Select batches randomly
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=4
    )
    
    # test
    test_dataset = dataset.BertDataset(
        text = test_df.text.values,
        label = test_df.label.values
    )
    test_dataloader = DataLoader(
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

    model = BertForSequenceClassification.from_pretrained(
               'bert-base-uncased', 
               num_labels = 2, # The number of output labels. 2 for binary classification.  
               output_attentions = False,
               output_hidden_states = False)
    model = model.to(device)
    #print(model.cuda())
    
    # calculate the number of training steps
    # this is used by scheduler
    num_train_steps = int(len(train_df) / config.TRAIN_BATCH_SIZE * config.EPOCHS)

    # AdamW optimizer, the most widely used optimzer for transformer based networks
    optimizer = AdamW(model.parameters(),
                           lr=config.LEARNING_RATE, # args.learning_rate - default is 5e-5
                           eps = 1e-8 # args.adam_epsilon  - default is 1e-8
                          )
    

    # fetch a scheduler
    # you can also try using reduce lr on plateau
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0, # Default value in run_glue.py
        num_training_steps=num_train_steps
    )

    # if you have multiple GPUs, model model to DataParallel to use multiple GPUs
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    # Set the seed value all over the place to make this reproducible.
    seed_val = 66

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # We'll store a number of quantities such as training and validation loss, validation accuracy, and timings.
    training_stats = []
    
    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # start training the epochs
    for epoch in range(config.EPOCHS):
        
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, config.EPOCHS))
        
        # Measure how long the training epoch takes.
        t0 = time.time()
        
        # Reset the total loss for this epoch.
        total_train_loss = 0
        
        # Put the model into training mode
        model.train()
        
        # For each batch of training data
        for step, batch in enumerate(train_dataloader):
            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
                
            # extract ids, token type ids and mask from current batch, also extract targets
            ids = batch["ids"]
            mask = batch["mask"]
            labels = batch["labels"]

            # move everything to specified device
            b_input_ids = ids.to(device,dtype=torch.long)
            b_input_mask = mask.to(device,dtype=torch.long)
            b_labels = labels.to(device,dtype=torch.long)
            
            model.zero_grad() 
            outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask, 
                            labels=b_labels)

            loss= outputs[0]
            logits = outputs[1]
            total_train_loss += loss.item()
            
            # Perform a backward pass to calculate the gradients.
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            # Update the learning rate.
            scheduler.step()
        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)            
    
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)  

        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))
        
        #  Validation
        # After the completion of each training epoch, measure our performance on validation set.
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables 
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in test_dataloader:

            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using 
            # the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            # extract ids, token type ids and mask from current batch, also extract targets
            ids = batch["ids"]
            mask = batch["mask"]
            labels = batch["labels"]

            # move everything to specified device
            b_input_ids = ids.to(device,dtype=torch.long)
            b_input_mask = mask.to(device,dtype=torch.long)
            b_labels = labels.to(device,dtype=torch.long)
            
            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():        

                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which 
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like the softmax.
                outputs= model(b_input_ids,
                               token_type_ids=None,
                               attention_mask=b_input_mask,
                               labels=b_labels)

            # Accumulate the validation loss.
            loss= outputs[0]
            logits = outputs[1] 

            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += flat_accuracy(logits, label_ids)


        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(test_dataloader)

        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))     
    
    
    # Display floats with two decimal places.
    pd.set_option('precision', 2)

    # Create a DataFrame from our training statistics.
    df_stats = pd.DataFrame(data=training_stats)

    # Use the 'epoch' as the row index.
    df_stats = df_stats.set_index('epoch')

    # Display the table.
    print(df_stats)
    
    output_dir = config.output_dir

    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer = config.TOKENIZER
    tokenizer.save_pretrained(output_dir)
    

def predict_finc():
    
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
#         print('predictions: ',len(predictions))
        
    print('    DONE.')    
    
    # # Combine the results across all batches. 
    flat_predictions = np.concatenate(predictions, axis=0)

    # # For each sample, pick the label (0 or 1) with the higher score.
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten() 
    acc = metrics.accuracy_score(true_labels, flat_predictions)
    print('predicte acc: ',acc)


if __name__ == "__main__":
    train()


    
    

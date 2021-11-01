import pandas as pd
import numpy as np
import random
import json
import csv
import time
import datetime
import os


import torch
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, random_split
from transformers import BertForSequenceClassification, AdamW, BertConfig

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


hierarchy = {}
label_to_id = {}

def types_to_specific_type(type_list):
    if len(type_list) == 0:
        return None 
    return type_list[0]


#def types_to_top_type(type_list):
#    if len(type_list) == 0:
#        return None 
#    return type_list[-1]
def types_to_top_type(type_list):
    generic = type_list[-1]
    for anst in type_list:
        for top_t in label_to_id:
            if anst == top_t or anst in  hierarchy[top_t]['children']:
                  return  top_t
    return generic

def get_hierarchy(dbpedia_types, hierarchy_json):

    with open(hierarchy_json) as json_file:
        hierarchy = json.load(json_file)
        
    for i,row in dbpedia_types.iterrows():
        parent = row['Parent']
        child = row['Type']
        if parent not in hierarchy:
            hierarchy[parent] ={}
            hierarchy[parent]['children'] =[]
        hierarchy[parent]['children'].append(child)

    hierarchy['dbo:Location'] = hierarchy['dbo:Place']
    hierarchy['dbo:Location']['children'].append('dbo:Place')
    hierarchy['dbo:MedicalSpecialty'] = {'children':['dbo:MedicalSpecialty']}
    hierarchy['dbo:PublicService'] = {'children':['dbo:PublicService']}
    
    return hierarchy

def get_label_and_id(mapping_csv):
    label_to_id = {}
    id_to_label = {}
    with open(mapping_csv) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            label_to_id[row[0]] = row[1]
            id_to_label[row[1]] = row[0]
            
    return label_to_id,id_to_label

if __name__ == '__main__':

    # training data from the challenge 2021
    train_fname = '../inputs/dataset/smart2021-AT_Answer_Type_Prediction/dbpedia/task1_dbpedia_train.json'
    dbpedia_df = pd.read_json(train_fname)
    
    # split dataset
    dbpedia_train_df = dbpedia_df.sample(frac=0.9,random_state=0,axis=0)
    dbpedia_test_df = dbpedia_df[~dbpedia_df.index.isin(dbpedia_train_df.index)]
    print('train and val size: ',len(dbpedia_train_df))
    print('test size:',len(dbpedia_test_df))
    dbpedia_test_df.to_json('../inputs/2021_dbpedia_0.1test.json', orient='records')
    
    # cleaning DBpedia dataset
    dbpedia_train_df= dbpedia_train_df[dbpedia_train_df.category.notna()]
    dbpedia_train_df= dbpedia_train_df[dbpedia_train_df['type'].notna()]
    dbpedia_train_df.dropna(subset=['question'], inplace=True)

    """Load models &  resources"""
    resources_dir = '../inputs/resources_dir' 
    mapping_csv = resources_dir+'/mapping.csv'
    hierarchy_json = resources_dir+'/dbpedia_hierarchy.json'
    
    dbpedia_types = pd.read_csv('../inputs/dbpedia_types.tsv', sep='\t')
    hierarchy = get_hierarchy(dbpedia_types, hierarchy_json)
    
    label_to_id,id_to_label = get_label_and_id(mapping_csv)
    
    dbpedia_train_df['top_level_type'] =dbpedia_train_df.type.apply(types_to_top_type)
    dbpedia_train_df= dbpedia_train_df[dbpedia_train_df['top_level_type'].notna()]
    
    # also added type order from the training set
    dbpedia_train_resource_df = dbpedia_train_df[dbpedia_train_df.category == 'resource']
    type_list = dbpedia_train_resource_df.type.values
        
    # Instantiate the Bert tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    input_ids = []
    attention_masks = []
    labels = []

    MAX_LENGTH = 36

    # For every sentence...
    for i,row in dbpedia_train_resource_df.iterrows():
        # sent = str('[CLS]') + row['question'] + str('[SEP]')
        sent = row['question']
        if row['top_level_type'] not in label_to_id: continue

        labels.append(int(label_to_id[row['top_level_type']]))

        encoded_dict = tokenizer.encode_plus(
                            sent,                         # Sentence to encode.
                            add_special_tokens = True,    # Add '[CLS]' and '[SEP]'
                            max_length = MAX_LENGTH,      # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True, # Construct attn. masks.
                            return_tensors = 'pt',        # Return pytorch tensors.
                       )

        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    print('label shape:',labels.shape)

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)

    # Create a 90-10 train-validation split.
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))

    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # The DataLoader needs to know batch size for training.
    batch_size = 32

    # Create the DataLoaders for training and validation sets.
    train_dataloader = DataLoader(
                train_dataset,  # The training samples.
                sampler = RandomSampler(train_dataset), # Select batches randomly
                batch_size = batch_size # Trains with this batch size.
            )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
                val_dataset, # The validation samples.
                sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                batch_size = batch_size # Evaluate with this batch size.
            )


    model = BertForSequenceClassification.from_pretrained(
                   'bert-base-uncased', 
                   num_labels = len(label_to_id), # The number of output labels. 2 for binary classification.  
                   output_attentions = False,
                   output_hidden_states = False)   

    # Tell pytorch to run this model on the GPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # send model to device
    model.to(device)  

    optimizer = AdamW(model.parameters(),
                      lr = 5e-5, # args.learning_rate - default is 5e-5
                      eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                    )

    # Number of training epochs. The BERT authors recommend between 2 and 4. 
    # We chose to run for 2,I have already seen that the model starts overfitting beyound 2 epochs
    epochs = 5

    # Total number of training steps is [number of batches] x [number of epochs]. 
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                           num_warmup_steps = 0, # Default value in run_glue.py
                           num_training_steps = total_steps)

    # Set the seed value all over the place to make this reproducible.
    seed_val = 66

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # store quantities such as training and validation loss, validation accuracy, and timings.
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # For each epoch...
    for epoch_i in range(0, epochs):

        # training
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode.
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack this training batch from dataloader and copy each tensor to the GPU.
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Always clear any previously calculated gradients before performing a backward pass. 
            model.zero_grad()        

            # Perform a forward pass (evaluate the model on this training batch).
            outputs = model(b_input_ids, 
                    token_type_ids=None, 
                    attention_mask=b_input_mask, 
                    labels=b_labels)

            # Accumulate the training loss over all of the batches. 
            loss= outputs[0]
            logits = outputs[1]  
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)            

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.4f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables 
        total_eval_accuracy = 0.0
        total_eval_loss = 0.0
        nb_eval_steps = 0.0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            with torch.no_grad():        

                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which 
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like the softmax.
                outputs= model(
                                b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask,
                                labels=b_labels
                            )

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
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.4f}".format(avg_val_accuracy))

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)

        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.4f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

    # Display floats with two decimal places.
    pd.set_option('precision', 4)

    # Create a DataFrame from our training statistics.
    df_stats = pd.DataFrame(data=training_stats)

    # Use the 'epoch' as the row index.
    df_stats = df_stats.set_index('epoch')

    output_dir = '../outputs/models_dbpedia/'
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df_stats.to_csv(output_dir + 'df_resources_top_stats.csv',index=None)

    models_dir = output_dir + 'BERT_resources_top/'

    # Create output directory if needed
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)


    print("Saving model to %s" % models_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(models_dir)
    tokenizer.save_pretrained(models_dir)











    
    

# data processing.py

import numpy as np
import pandas as pd
import json
import csv
import os

if __name__ == "__main__":
    # DBPedia hierarchy provided by the challenge organizers
    
    # train dataset file
    dataset_path = '../inputs/datasets/'
    dbpedia_train_resampled = 'output/dbpedia_train_resampled.csv'
    dbpedia_valid_resampled = 'output/dbpedia_valid_resampled.csv'
    
    dbpedia_train_resampled = os.path.join(dataset_path, dbpedia_train_resampled)
    dbpedia_valid_resampled = os.path.join(dataset_path, dbpedia_valid_resampled)
    dbpedia_types = os.path.join(dataset_path,'dbpedia_types.tsv')
    
    dbpedia_types = pd.read_csv(dbpedia_types, sep='\t')
    train_resampled = pd.read_csv(dbpedia_train_resampled)
    valid_resampled = pd.read_csv(dbpedia_valid_resampled)
    
    # training data frm the challenge
    dbpedia_df = pd.read_json('../inputs/datasets/DBpedia/smarttask_dbpedia_train.json')
    
    # also added type order from the training set
    dbpedia_res_df = dbpedia_df[dbpedia_df.category == 'resource']
    type_list = dbpedia_res_df.type.values
    
    test_df= pd.read_json('../inputs/datasets/DBpedia/smarttask_dbpedia_test.json')
    
    # cleaning DBpedia dataset
    dbpedia_df= dbpedia_df[dbpedia_df.category.notna()]
    test_df= test_df[test_df.category.notna()]

    dbpedia_df= dbpedia_df[dbpedia_df['type'].notna()]
    test_df= test_df[test_df['type'].notna()]

    dbpedia_df.dropna( subset=['question'], inplace=True)
    test_df.dropna( subset=['question'], inplace=True)
    
    # only choose sample with 'resource' category
    dbpedia_res_df = dbpedia_df[dbpedia_df.category == 'resource']
    test_res_df = test_df[test_df.category == 'resource']

    id2question ={}
    for i, row in dbpedia_res_df.iterrows():
        id2question[row['id']]= row['question']
    for i, row in test_res_df.iterrows():
        id2question[row['id']]= row['question']

    sent_train_list = []    
    for i,row in train_resampled.iterrows():
        sent =id2question[row['id']]+'[SEP]'+row['specific_type']
        sent_train_list.append(sent)
   
    sent_test_list = []    
    for i,row in train_resampled.iterrows():
        sent =id2question[row['id']]+'[SEP]'+row['specific_type']
        sent_test_list.append(sent)        
    
        
        
    train_resampled['text'] = pd.DataFrame(sent_train_list)
    valid_resampled['text'] = pd.DataFrame(sent_test_list)
    # only save text and class
    train_resampled_final = train_resampled.loc[:,['text','class']]
    valid_resampled_final = valid_resampled.loc[:,['text','class']]
    
    train_resampled_final.rename(columns={'class':'label'},inplace=True)
    valid_resampled_final.rename(columns={'class':'label'},inplace=True)
    
    # save to csv file
    train_resampled_final.to_csv(os.path.join(dataset_path,'train_resampled_final.csv'),index=False,)
    valid_resampled_final.to_csv(os.path.join(dataset_path,'valid_resampled_final.csv'),index=False,)
    
    
  



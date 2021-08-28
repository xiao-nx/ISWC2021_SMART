# model.py

import config 
import transformers
import torch.nn as nn 

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # we fetch the model from the BERT_PATH defined in config.py
        self.bert_embedding = transformers.BertModel.from_pretrained(
            'bert-base-uncased'
        )
        
        # add a dropout for regularization
        self.bert_drop = nn.Dropout(config.DROPOUT_RATE)
        
        # add a fc layer
        embed_dim = 768   # bert embedding dim is 768
        hidden_dim = 300
        
        self.fc1 = nn.Sequential(
                            nn.Linear(embed_dim, hidden_dim), 
                            nn.BatchNorm1d(hidden_dim), 
                            nn.ReLU(True))
        self.fc2 = nn.Linear(hidden_dim, config.NUM_CLASS)


    def forward(self, ids, mask, token_type_ids):
        # BERT in its default settings returns two outputs
        # last hidden state and output of bert pooler layer
        # output of the pooler size (batch_size, hidden_size)
        # hidden size can be 768 or 1024 depending on if using bert base or large respectively
        _, bert_output = self.bert_embedding(
            ids,
            attention_mask = mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )
        
        # pass through dropout layer
        bert_output = self.bert_drop(bert_output)

        # pass through linear layer
        fc1_output = self.fc1(bert_output)
        logits = self.fc2(fc1_output)

        return logits
    
    
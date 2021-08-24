# https://zhuanlan.zhihu.com/p/73176084

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, vocab_size, embedding_pretrained=None):
        super(Model, self).__init__()
        
        self.n_gram_vocab = 250499 ## 自定义大小
        self.embed_size = embedding_pretrained.size(1) if embedding_pretrained is not None else 300
        self.dropout_p = 0.3 
        self.hidden_size = 256  
        self.num_classes = 2
        
        # Embedding        
        self.text_embedding = nn.Embedding(
                                num_embeddings=vocab_size,
                                embedding_dim=self.embed_size)
        self.entity_embedding = nn.Embedding(
                                num_embeddings=vocab_size,
                                embedding_dim=self.embed_size)
        
        if embedding_pretrained is not None:
            self.text_embedding.weight.data.copy_(embedding_pretrained) #load pretrained
            self.entity_embedding.weight.data.copy_(embedding_pretrained)
        
        # whether train the pretrained embeddings
        self.text_embedding.weight.requires_grad = False
        self.entity_embedding.weight.requires_grad = False
        
        self.embedding_ngram2 = nn.Embedding(self.n_gram_vocab, self.embed_size)
        self.embedding_ngram3 = nn.Embedding(self.n_gram_vocab, self.embed_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.fc1 = nn.Linear(self.embed_size * 4, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        """
        x: (sentence_length, batch)
        """
        batch_text = x['text']
        batch_entity = x['entity1']
        
        batch_text = batch_text.permute(1,0) # x: [batch, sentence_length]
        batch_entity = batch_entity.permute(1,0)

        text_embs = self.text_embedding(batch_text)
        entity_embs = self.entity_embedding(batch_entity)   
        input_embs = torch.cat([entity_embs, text_embs],dim=2)        


        #out_word = self.embedding(x) # x: [batch_size, seq_len, embed_size]
        out_bigram = self.embedding_ngram2(batch_text) # x: [batch_size, seq_len, embed_size]
        out_trigram = self.embedding_ngram3(batch_text) # x: [batch_size, seq_len, embed_size]
        
        out = torch.cat((input_embs, out_bigram, out_trigram), -1)

        out = out.mean(dim=1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = F.relu(out)
        fc2 = self.fc2(out)
        
        logit = F.log_softmax(fc2, dim=1)
        
        return logit
    
    
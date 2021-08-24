# textcnn.py 

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, vocab_size, embedding_pretrained=None):
        '''
        A CNN for text classification.
        Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
        '''
        super(Model, self).__init__()
        
        self.max_length = 512
        self.embed_size = embedding_pretrained.size(1) if embedding_pretrained is not None else 50
        self.kernel_size = [3, 4, 5]
        self.kernel_num = 16
        self.dropout_p = 0.3
        self.num_class = 2

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
        
        # a simple CNN
        self.conv1 = nn.Conv2d(1, self.kernel_num, (self.kernel_size[0], self.embed_size * 2))
        self.conv2 = nn.Conv2d(1, self.kernel_num, (self.kernel_size[1], self.embed_size * 2))
        self.conv3 = nn.Conv2d(1, self.kernel_num, (self.kernel_size[2], self.embed_size * 2))
        
        self.dropout = nn.Dropout(self.dropout_p)

        self.fc = nn.Linear(len(self.kernel_size) * self.kernel_num, self.num_class)
        
    @staticmethod
    def conv_and_pool(x, conv):
        # x: (batch, 1, sentence_length)
        x = conv(x)
        # x: (batch, kernel_num, H_out, 1)
        x = F.relu(x.squeeze(3))
        # x: (batch, kernel_num, H_out)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        #  (batch, kernel_num)
        return x

    def forward(self, x):
        """
        x: (batch, sentence_length)
        """
        
        batch_text = x['text']
        batch_entity = x['entity1']
        
        batch_text = batch_text.permute(1,0) # x: [batch, sentence_length]
        batch_entity = batch_entity.permute(1,0)

        text_embs = self.text_embedding(batch_text)
        entity_embs = self.entity_embedding(batch_entity)
        
        input_embs = torch.cat([entity_embs, text_embs],dim=2)
        x = input_embs.unsqueeze(1)
            
        # x: (batch, 1, sentence_length, embed_dim)
        x1 = self.conv_and_pool(x, self.conv1)  # (batch, kernel_num)
        x2 = self.conv_and_pool(x, self.conv1)  # (batch, kernel_num)
        x3 = self.conv_and_pool(x, self.conv1)  # (batch, kernel_num)
        x = torch.cat((x1, x2, x3), 1)  # (batch, 3 * kernel_num)
        x = self.dropout(x)
        
        logit = F.log_softmax(self.fc(x), dim=1)
        
        #logit = torch.sigmoid(self.fc(x))
        #logit = self.fc(x)
        
        return logit       
        

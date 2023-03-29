import random

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers=1, dropout_p=0.5):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # self.dropout = nn.Dropout(dropout_p)
        
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.GRU(embedding_size, hidden_size, num_layers, dropout = dropout_p, batch_first=True)
        
        
    def forward(self, x): # x.shape = [batch size, input len]
        
        # embedded = self.dropout(self.embedding(x))
        embedded = self.embedding(x) # [batch size, input len, emb dim]
        
        outputs, hidden = self.rnn(embedded)
        
        # outputs = [batch size, x len, hidden size * num direction] / num direction = 1
        # hidden = [n layers * num direction, batch size, hidden size] / num direction = 1
        
        return hidden


class Decoder(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, num_layers=1, dropout_p=0.5):
        super().__init__()
        
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(output_size, embedding_size)
        # self.dropout = nn.Dropout(dropout_p)
        
        self.rnn = nn.GRU(embedding_size, hidden_size, num_layers, dropout = dropout_p, batch_first=True)
        
        self.fc_out = nn.Linear(hidden_size, output_size)
        
        
    def forward(self, input, hidden):
        # input.shape = [batch size] // list of tokens, len of batch size
        # hidden = [n layers (* n directions), batch size, hidden size] / num direction = 1
        
        input = input.unsqueeze(1) # [batch size, 1]
        
        # embedded = self.dropout(self.embedding(input))
        embedded = self.embedding(input)
        
        # embedded = [batch size, 1, emb size]
                
        output, hidden = self.rnn(embedded, hidden)
        
        # num direction always 1
        # outputs = [batch size, 1, hidden size]
        # hidden = [n layers, batch size, hidden size]
        
        prediction = self.fc_out(output.squeeze(1))
        
        # prediction = [batch size, output size]
        
        return prediction, hidden
    
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device='cpu'):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hidden_size == decoder.hidden_size, "Hidden size not matched"
        assert encoder.num_layers == decoder.num_layers, "Number of layers not matched"
        
    def forward(self, input, target, teacher_forcing_ratio = 0.5):
        
        # input = [batch size, src len]
        # target = [batch size, target len]
        # teacher_forcing_ratio is probability to use teacher forcing
        target_len = target.shape[1]
        
        # tensor to store decoder outputs
        outputs = [] # expected output shape: target_len, batch_size, target_vocab_size
        
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden = self.encoder(input)
        
        #first input to the decoder is the <sos> tokens
        decoder_input = target[:, 0]
        
        for t in range(1, target_len):
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden= self.decoder(decoder_input, hidden)
            
            #place predictions in a tensor holding predictions for each token
            outputs.append(output) # 0 ~ target_len - 2
            
            #get the highest predicted token from our predictions
            top_pred = output.argmax(1)
            
            teacher_force = random.random() < teacher_forcing_ratio
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            decoder_input = target[:, t] if teacher_force else top_pred
        
        return torch.stack(outputs) # [target len - 1, batch size, target vocab size]
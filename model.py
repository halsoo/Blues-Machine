import random

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers=1, dropout_p=0.5, device='cpu'):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.device = device
        
        self.embedding = nn.Embedding(input_size, embedding_size).to(self.device)
        self.rnn = nn.GRU(embedding_size, hidden_size, num_layers, dropout = dropout_p, batch_first=True).to(self.device)
        
        
    def forward(self, x): 
        # x: (batch size, input len)
        embedded = self.embedding(x) # (batch size, input len, emb dim)
        
        _, hidden = self.rnn(embedded)
        
        # outputs: (batch size, input len, hidden size * num dir)
        # hidden: (n layers, batch size, hidden size)
        
        return hidden


class Decoder(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, num_layers=1, dropout_p=0.5, device='cpu'):
        super().__init__()
        
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.device = device
        
        self.embedding = nn.Embedding(output_size, embedding_size).to(self.device)
        
        self.rnn = nn.GRU(embedding_size, hidden_size, num_layers, dropout = dropout_p, batch_first=True).to(self.device)
        
        self.fc_out = nn.Linear(hidden_size, output_size).to(self.device)
        
        
    def forward(self, input, hidden):
        # input: (batch size, input len)
        # hidden: (n layers, batch size, hidden size)

        embedded = self.embedding(input) # (batch size, input len, emb size)
                
        output, hidden = self.rnn(embedded, hidden)
        
        # output: (batch size, input len, hidden size)
        # hidden: (n layers, batch size, hidden size)
        
        prediction = self.fc_out(output)
        
        # prediction = (batch size, input len, output size)
        
        return prediction, hidden
    
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device='cpu'):
        super().__init__()
        self.device = device

        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        
        assert encoder.hidden_size == decoder.hidden_size, "Hidden size not matched"
        assert encoder.num_layers == decoder.num_layers, "Number of layers not matched"
        
    def forward(self, input, target, teacher_forcing_ratio = 0.5, parallel_calc = True):
        
        # input: (batch size, src len)
        # target: (batch size, target len)
        # teacher_forcing_ratio: probability of teacher forcing
        
        # expected output shape: (target_len-1, batch_size, target_vocab_size)
        # target_len - 1: except <sos>
        
        input.to(self.device)
        target.to(self.device)
        
        hidden = self.encoder(input)
        
        if parallel_calc:
            outputs, _ = self.decoder(target[:, :-1], hidden)
            return outputs
            
        else:
            outputs = []
            target_len = target.shape[1]
        
            #<sos> tokens
            decoder_input = target[:, 0].unsqueeze(1).to(self.device) # (batch size) => (batch size, 1)
            
            for t in range(1, target_len): # after <sos>, before <eos>
                preds, hidden= self.decoder(decoder_input, hidden)
                
                #place predictions in a tensor holding predictions for each token
                outputs.append(preds)
                
                #get the highest predicted token from our predictions
                top_pred = preds.softmax(2).argmax(2).to(self.device)
                
                teacher_force = random.random() < teacher_forcing_ratio
                
                decoder_input = target[:, t].unsqueeze(1).to(self.device) if teacher_force else top_pred
                
            outputs = torch.stack(outputs).squeeze(2).to(self.device)
            outputs = outputs.permute(1, 0, 2).contiguous().to(self.device) # (batch size, target len - 1, target vocab size)
            
            return outputs
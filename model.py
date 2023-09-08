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
      
  def forward(self, input, target, inference_mode=False, max_len = None):
        
    # input: (batch size, src len)
    # target: (batch size, target len)
    
    # expected output shape: (target_len-1, batch_size, target_vocab_size)
    # target_len - 1: except <sos>
    
    input.to(self.device)
    target.to(self.device)
    
    hidden = self.encoder(input)
    
    if inference_mode:
      outputs = []

      decoder_input = target # <sos> token
      
      for _ in range(max_len): # max_len = same as input's number of bars
          preds, hidden= self.decoder(decoder_input, hidden) # pred = (batch size=1, input len=1, vocab size)
          outputs.append(preds)
          decoder_input = preds.squeeze(1).softmax(1).argmax(1).to(self.device)
          
      outputs = torch.stack(outputs).squeeze(1).to(self.device) # (max len, 1, vocab size) => (max len, vocab size)
      
      return outputs
        
    else:
      outputs, _ = self.decoder(target[:, :-1], hidden)
      return outputs  
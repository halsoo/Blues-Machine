import time
import pickle

from tqdm import tqdm

import torch

class Trainer:
    def __init__(self, model, optimizer, loss_fn, train_loader, valid_loader, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        
        self.device = device
        self.model.to(self.device)
        self.loss_fn.to(self.device)
        
        self.best_valid_accuracy = 0
        
        self.training_loss = []
        self.validation_loss = []
        self.validation_acc = []

    def save_model(self, path='models/', surfix=''):
        model_default_name = 'blues_call_and_res_model-'
        torch.save({'model':self.model.state_dict(), 'optim':self.optimizer.state_dict()}, f'{path}{model_default_name}{int(time.time())}{surfix}.pt')
        
    def _get_accuracy(self, pred, target_sqz):
        # pred: [batch size * pred len, num vocab]
        # target_sqz: [batch size * target len]
        
        is_correct = pred.argmax(dim=-1).to(self.device) == target_sqz
        return (is_correct.sum() / target_sqz.reshape(-1,).shape[-1]).item()
        
    def train_by_num_epoch(self, num_epochs, tfr=0.5, do_validate=False): # tfr: teacher forcing ratio
        for epoch in tqdm(range(num_epochs)):
            self.model.train()
            
            for batch in self.train_loader:
                loss_value = self._train_by_single_batch(batch, tfr=tfr)
                self.training_loss.append(loss_value)
            
            if do_validate:
                self.model.eval()
                
                validation_loss, validation_acc = self.validate()
                self.validation_loss.append(validation_loss)
                self.validation_acc.append(validation_acc)
                
                if validation_acc > self.best_valid_accuracy:
                    print(f"Saving the model with best validation accuracy: Epoch {epoch+1}, Acc: {validation_acc:.4f} ")
                    self.save_model()
                # else:
                #     self.save_model()
                    
                self.best_valid_accuracy = max(validation_acc, self.best_valid_accuracy)

        
    def _train_by_single_batch(self, batch, tfr=0.5):
        self.model.train()

        call, res = batch
        call.to(self.device)
        res.to(self.device)
        output = self.model(call, res, teacher_forcing_ratio=tfr) # (batch size, target len, target vocab size)
        
        vocab_size = output.shape[-1]
        
        res = res[:, 1:].contiguous().view(-1)
        output = output.view(-1, vocab_size)
        
        loss = self.loss_fn(output, res)
        
        self.optimizer.zero_grad()
        
        loss.backward()
        
        self.optimizer.step()
        
        return loss.item()


    def validate(self, external_loader=None):
        
        if external_loader and isinstance(external_loader, DataLoader):
            loader = external_loader
            print('An arbitrary loader is used instead of Validation loader')
        else:
            loader = self.valid_loader
        
        self.model.eval()
        
        with torch.no_grad():
            valid_loss = 0
            valid_acc = 0
            batch_cnt = 0
            
            for batch in loader:
                call, res = batch
                
                output = self.model(call, res, teacher_forcing_ratio=1.0) # (batch size, target len, target vocab size)
                
                vocab_size = output.shape[-1]
                
                res = res[:, 1:].contiguous().view(-1)
                output = output.view(-1, vocab_size)
                
                loss = self.loss_fn(output, res)
                
                valid_loss += loss.item()
                valid_acc += self._get_accuracy(output, res)
                
                batch_cnt += 1
                
            valid_loss /= batch_cnt
            valid_acc /= batch_cnt
                
            return valid_loss, valid_acc


def load_loss(date, num):
    try:
        with open(f"models/blues_call_and_res_model-loss-{date}-{num}.p", 'rb') as f: 
            return pickle.load(f)
    except FileNotFoundError:
        return {}


def save_loss(date, num, loss_obj): # loss obj = { train_loss, valid_loss }
    with open(f"models/blues_call_and_res_model-loss-{date}-{num}.p", 'wb') as f:
        pickle.dump(loss_obj, f)
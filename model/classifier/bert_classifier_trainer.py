import torch
from torch import Tensor as T
from torch.utils.data import DataLoader

from .bert_classifier import BertClassifier

class BertClassifierTrainer():
    def __init__(self,
                 classifier: BertClassifier=None,
                 batch_size: int=64) -> None:
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.batch_size = batch_size
        
        self.classifier = classifier.to(self.device)
        
    def train(self, 
              train_data, 
              shuffle: bool=True, 
              max_epoch: int=10,
              loss_func_type: str=None,
              optimizer_type : str=None,
              learning_rate: float=0.001):
        
        self.train_data = train_data
        train_dataloader = DataLoader(self.train_data,
                                      batch_size=self.batch_size,
                                      shuffle=shuffle,
                                      num_workers=2)
        
        size = len(self.train_data)
        
        self.classifier.train()
        
        self.loss_func_type = loss_func_type if loss_func_type else ""
        loss_func = self.select_loss_func(self.loss_func_type)
        
                # initialize optimizer
        self.optimizer = optimizer_type if optimizer_type else self.optimizer
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        train_loss_history = []
        for epoch in range(max_epoch):
            
            print(f"Epoch {epoch+1}\n-------------------------------")
            
            batch_loss = []
            for index_batch, sample_batch in enumerate(train_dataloader):
                
                logit = self.classifier(sample_batch)
                
                loss = loss_func(logit)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if index_batch % 10 == 0:
                    current = (index_batch + 1) * len(sample_batch)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                    batch_loss.append(float(loss))
                
            train_loss_history.append(batch_loss)
            print()
        
        self.train_loss_history = train_loss_history
        print("Bert Classifier Training Complete!")
        return(train_loss_history)
                
    def select_loss_func(self, loss_func_type: str):
        
        if loss_func_type == 'cross_entropy':
            pass
        else:
            pass
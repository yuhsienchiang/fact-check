import torch
from torch import Tensor as T
from torch import nn
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
              train_dataset, 
              shuffle: bool=True, 
              max_epoch: int=10,
              loss_func_type: str=None,
              optimizer_type : str=None,
              learning_rate: float=0.001):
        
        self.train_dataset = train_dataset
        train_dataloader = DataLoader(self.train_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=shuffle,
                                      collate_fn=self.train_dataset.train_collate_fn,
                                      num_workers=2)
        
        size = len(self.train_dataset)
        
        self.classifier.train()
        
        self.loss_func_type = loss_func_type
        loss_func = self.select_loss_func(self.loss_func_type)
        
        # initialize optimizer
        self.optimizer_type = optimizer_type
        if self.optimizer_type == "adam":
            optimizer = torch.optim.Adam(self.classifier.parameters(), lr=learning_rate)
        elif self.optimizer_type == "SGD":
            optimizer = torch.optim.SGD(self.classifier.parameters(), lr=learning_rate)
        else:
            optimizer = torch.optim.Adam(self.classifier.parameters(), lr=learning_rate)
        
        train_loss_history = []
        for epoch in range(max_epoch):
            
            print(f"Epoch {epoch+1}\n-------------------------------")
            
            batch_loss = []
            for index_batch, sample_batch in enumerate(train_dataloader):
                
                text_sequences = sample_batch.text_sequence
                labels = sample_batch.query_label.to(self.device)
                
                logit = self.classifier(input_ids=text_sequences.input_ids.to(self.device),
                                        token_type_ids=text_sequences.segments.to(self.device),
                                        attention_mask=text_sequences.attn_mask.to(self.device),
                                        return_dict=True)
                
                logit = torch.flatten(logit, 0, 1).to(self.device)
                
                loss = loss_func(logit, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if index_batch % 10 == 0:
                    current = (index_batch + 1) * self.batch_size
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                    batch_loss.append(float(loss))
                elif len(labels) < self.batch_size:
                    current = size
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                    batch_loss.append(float(loss))
                
                del logit, loss, labels
                
            train_loss_history.append(batch_loss)
            print()
        
        self.train_loss_history = train_loss_history
        print("Bert Classifier Training Complete!")
        return(train_loss_history)


    def select_loss_func(self, loss_func_type: str):
        
        if loss_func_type == 'cross_entropy':
            return nn.CrossEntropyLoss()
        else:
            pass
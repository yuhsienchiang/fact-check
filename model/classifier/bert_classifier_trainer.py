import torch
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
              learning_rate: float=1e-5):
        
        self.train_dataset = train_dataset
        self.shuffle = shuffle
        
        train_dataloader = DataLoader(self.train_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=self.shuffle,
                                      num_workers=4,
                                      collate_fn=self.train_dataset.train_collate_fn)
        
        size = len(self.train_dataset)
        
        self.classifier.train()
        
        self.loss_func_type = loss_func_type if loss_func_type else "cross_entroy"
        loss_func = self.select_loss_func(self.loss_func_type)
        
        # initialize optimizer
        self.optimizer_type = optimizer_type if optimizer_type else "adam"
        if self.optimizer_type == "adam":
            optimizer = torch.optim.Adam(self.classifier.parameters(), lr=learning_rate)
        elif self.optimizer_type == "SGD":
            optimizer = torch.optim.SGD(self.classifier.parameters(), lr=learning_rate)
        
        self.train_loss_history = []
        for epoch in range(max_epoch):
            
            print(f"Epoch {epoch+1}\n-------------------------------")
            
            batch_loss = []
            batch_trained_sample = 0
            for index_batch, sample_batch in enumerate(train_dataloader):
                
                text_sequences = sample_batch.text_sequence
                text_sequences_input_ids = text_sequences.input_ids.squeeze(1).to(self.device) 
                text_sequences_token_type_ids = text_sequences.segments.squeeze(1).to(self.device) 
                text_sequences_attention_mask = text_sequences.attn_mask.squeeze(1).to(self.device) 
                labels = sample_batch.query_label.to(self.device)
                
                logit = self.classifier(input_ids=text_sequences_input_ids,
                                        token_type_ids=text_sequences_token_type_ids,
                                        attention_mask=text_sequences_attention_mask,
                                        return_dict=True)
                
                loss = loss_func(logit, labels.to(torch.long)).to(self.device)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                batch_trained_sample += len(labels)
                if index_batch % 10 == 0 or batch_trained_sample == size:    
                    print(f"loss: {loss:>7f}  [{batch_trained_sample:>5d}/{size:>5d}]")
                    batch_loss.append(float(loss))
                
                del logit, loss, labels, text_sequences_input_ids, text_sequences_token_type_ids, text_sequences_attention_mask
                
            self.train_loss_history.append(batch_loss)
            print()
        
        print("Bert Classifier Training Complete!")
        return(self.train_loss_history)


    def select_loss_func(self, loss_func_type: str):
        
        if loss_func_type == 'cross_entropy':
            return nn.CrossEntropyLoss()
        else:
            pass
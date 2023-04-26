import torch
from torch import Tensor as T
from torch.utils.data import DataLoader
from torch.nn import functional as F

from transformers import BertTokenizer

from .biencoder import BiEncoder
from .bert_encoder import BertEncoder
from .biencoder_dataset import BiEncoderDataset



class BiEncoderTrainer():
    def __init__(self, 
                 model:BiEncoder=None, 
                 train_data :BiEncoderDataset=None,
                 tokenizer: BertTokenizer=None, 
                 optimizer=None, 
                 loss_func: str=None,
                 batch_size: int=64) -> None:
        
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.model = model
        self.train_data = train_data
        self.loss_func = loss_func
        self.batch_size = batch_size        
    
    def train(self, train_data: BiEncoderDataset=None, max_epoch: int=10):
        
        self.train_data = train_data if train_data is not None else self.train_data    
        train_dataloader = DataLoader(self.train_data, batch_size=self.batch_size)
        loss_func = self.select_loss_func("nll")
        
        size = len(self.train_data)
        
        self.model.train()
        
        for epoch in range(max_epoch):
            
            for index, sample_batch in enumerate(train_dataloader):
                
                query = sample_batch.query
                positive_evid = sample_batch.positive_evid
                negative_evid = sample_batch.negative_evid
                
                loss = loss_func(query_input_ids=query.input_ids,
                                 positive_evidence_input_ids=positive_evid.input_ids,
                                 negative_input_ids=negative_evid.input_ids)
                
                # backpropadation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            
    
    def select_loss_func(self, loss_func: str):
        return self.negative_likelihood_loss
    
    
    def negative_likelihood_loss(self, query_input_ids, positive_evidence_input_ids, negative_input_ids):
        
        positive_score = self.dot_similarity(query_input_ids, positive_evidence_input_ids)
        negative_score = self.dot_similarity(query_input_ids, negative_input_ids)
        
        log_softmax_score = F.log_softmax(torch.stack(positive_score, negative_score), dim=0)
        
        return F.nll_loss(log_softmax_score, target=[], reduction='mean')
        
        
    
    
    def dot_similarity(self, query_vec: T, evidence_vec: T):
        similarity = torch.matmul(query_vec, evidence_vec)
        return similarity
        
        
    
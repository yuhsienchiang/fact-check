import torch
from torch import Tensor as T
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .biencoder import BiEncoder
from .biencoder_dataset import BiEncoderDataset


class BiEncoderTrainer():
    def __init__(self, 
                 model:BiEncoder=None, 
                 batch_size: int=64) -> None:
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        
        self.model = model.to(self.device)
        self.batch_size = batch_size        

    
    def train(self, 
              train_data: BiEncoderDataset=None,
              shuffle: bool=True,
              max_epoch: int=10, 
              loss_func_type: str=None,
              similarity_func_type : str=None,
              optimizer_type: str=None, 
              learning_rate: float=0.001):
        
        # setup dataloader
        self.train_data = train_data
        train_dataloader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=shuffle, num_workers=2)
        size = len(self.train_data)
        
        # set model to training mode
        self.model.train()
        
        # initialize loss function
        self.similarity_func_type = similarity_func_type if similarity_func_type else "dot"
        self.loss_func_type = loss_func_type if loss_func_type else "nll_loss"
        
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
                
                query = sample_batch.query
                evid = sample_batch.evid
                
                # forward pass the input through the biencoder model 
                query_vector, evid_vector = self.model(query_ids=query.input_ids.to(self.device),
                                                       query_segment=query.segments.to(self.device),
                                                       query_attn_mask=query.attn_mask.to(self.device),
                                                       evid_ids=evid.input_ids.to(self.device),
                                                       evid_segment=evid.segments.to(self.device),
                                                       evid_attn_mask=evid.attn_mask.to(self.device))
                
                # calculate the loss
                loss = loss_func(query_vector=query_vector.to(self.device),
                                 evidence_vector=evid_vector.to(self.device))

                # backpropadation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # print info and store history
                if index_batch % 10 == 0:
                    current = (index_batch + 1) * len(sample_batch)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                    batch_loss.append(float(loss))
            
            train_loss_history.append(batch_loss)
            print()
        
        self.train_loss_history = train_loss_history
        print("Training Done!")
        return train_loss_history

    
    def select_loss_func(self, loss_func_type: str):
        
        if loss_func_type == 'nll_loss':
            return self.negative_likelihood_loss
        else:
            return None
        
    def select_similarity_func(self, similarity_func_type: str):
        if similarity_func_type == "dot":
            return self.dot_similarity
        elif similarity_func_type == "cosine":
            return self.cosine_similarity
    
    
    def dot_similarity(self, query_vec: T, evidence_vec: T):

        return torch.matmul(query_vec, evidence_vec.transpose(dim0=0, dim1=1))
    
    
    def cosine_similarity(self, query_vec: T, evidence_vec: T, eps: float=1e-8):
        
        dot_similarity = self.dot_similarity(query_vec=query_vec, evidence_vec=evidence_vec)
        
        query_norm = torch.unflatten(torch.linalg.vector_norm(query_vec, dim=1), 0, (-1, 1))
        evid_norm = torch.unflatten(torch.linalg.vector_norm(evidence_vec, dim=1), 0, (1, -1))
        
        norm_matrix = torch.matmul(query_norm, evid_norm)
        norm_matrix[norm_matrix < eps] = eps
        
        return torch.div(dot_similarity, norm_matrix)
    
    
    def negative_likelihood_loss(self, query_vector: T, evidence_vector: T):
        
        similarity_func = self.select_similarity_func(self.similarity_func_type)
        
        # ensure the vectors only has 2 dim
        if len(query_vector.shape) == 3:
            query_vector = query_vector.squeeze(1)
        
        if len(evidence_vector.shape) == 3:
            evidence_vector = torch.flatten(evidence_vector, 0, 1)
        
        similarity_score = similarity_func(query_vec=query_vector, evidence_vec=evidence_vector)
        
        num_query = len(query_vector)
        num_neg_evid = self.train_data.neg_evidence_num
        is_positive = torch.arange(0, (num_neg_evid + 1) * num_query, (num_neg_evid + 1)).to(self.device)
        
        log_softmax_score = F.log_softmax(similarity_score, dim=1)
        
        return F.nll_loss(log_softmax_score, target=is_positive, reduction="mean")    
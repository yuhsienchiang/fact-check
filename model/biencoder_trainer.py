import torch
from torch import Tensor as T
from torch.utils.data import DataLoader

from .biencoder import BiEncoder
from .biencoder_dataset import BiEncoderDataset


class BiEncoderTrainer():
    def __init__(self, 
                 model:BiEncoder=None, 
                 optimizer: str=None, 
                 loss_func: str=None,
                 batch_size: int=64) -> None:
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_func_type = loss_func
        self.batch_size = batch_size        

    
    def train(self, 
              train_data: BiEncoderDataset=None, 
              max_epoch: int=10, 
              loss_func_type: str=None, 
              optimizer_type: str=None, 
              learning_rate: float=0.001):
        
        # setup dataloader
        self.train_data = train_data
        train_dataloader = DataLoader(self.train_data, batch_size=self.batch_size)
        size = len(self.train_data)
        
        # set model to training mode
        self.model.train()
        
        # initialize loss function
        self.loss_func_type = loss_func_type if loss_func_type else self.loss_func_type
        loss_func = self.select_loss_func(self.loss_func_type) if self.loss_func_type else self.select_loss_func("nll_loss")
        
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
                                                       auery_attn_mask=query.attn_mask.to(self.device),
                                                       evid_ids=evid.input_ids.to(self.device),
                                                       evid_segment=evid.segments.to(self.device),
                                                       evid_attn_mask=evid.attn_mask.to(self.device))
                
                # calculate the loss
                loss = loss_func(query_vector=query_vector.to(self.device),
                                 evidence_vector=evid_vector.to(self.device),
                                 position_idx=evid.posit_neg_idx.to(self.device))

                # backpropadation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # print info and store history
                if index_batch % 10 == 0:
                    current = (index_batch + 1) * len(sample_batch)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                    batch_loss.append(loss)
            
            train_loss_history.append(batch_loss)
            print()
        
        self.train_loss_history = train_loss_history
        print("Training Done!")
        return train_loss_history

    
    def select_loss_func(self, loss_func: str):
        
        if loss_func == 'nll_loss':
            return self.negative_likelihood_loss
        else:
            return None
    
    
    def negative_likelihood_loss(self, query_vector, evidence_vector, position_idx):
        
        similarity_score = self.dot_similarity(query_vec=query_vector, evidence_vec=evidence_vector)
        shape = similarity_score.shape
        dimension = len(shape)
        
        expo_similarity_score = torch.exp(similarity_score)
        positive_mask = torch.zeros(shape)
        
        # 1. setting the scores of the padding vectors to zero
        # 2. creating the positive mask which indicates the positions the positive evidences
        for i in range(shape[0]):
            start = position_idx[i, 0]
            end = position_idx[i, 1]
            expo_similarity_score[i, start:end, :] = 0
            positive_mask[i, :start, :] = 1
            
        log_softmax_score = -torch.log(expo_similarity_score / expo_similarity_score.sum(dim=dimension-2).unsqueeze(dimension-2).expand(shape))
        log_softmax_score[log_softmax_score == torch.inf] = 0
        
        return (log_softmax_score * positive_mask).mean()
        
    
    def dot_similarity(self, query_vec: T, evidence_vec: T):
        query_dimension = len(query_vec.shape)
        return torch.matmul(evidence_vec, query_vec.transpose(dim0=query_dimension-2, dim1=query_dimension-1))

        
        
    
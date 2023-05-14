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
              train_dataset: BiEncoderDataset=None,
              shuffle: bool=True,
              max_epoch: int=10, 
              loss_func_type: str=None,
              optimizer_type: str=None, 
              learning_rate: float=2e-5):
        
        # setup dataloader
        self.shuffle = shuffle
        self.train_dataset = train_dataset
        
        train_dataloader = DataLoader(self.train_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=self.shuffle,
                                      num_workers=4,
                                      collate_fn=self.train_dataset.train_collate_fn)
        
        size = len(self.train_dataset)

        # set model to training mode
        self.model.train()
        self.model.encoder_0.train()
        if self.model.encoder_1 is not None:
            self.model.encoder_1.train()

        # initialize loss function
        self.loss_func_type = loss_func_type if loss_func_type else "nll_loss"
        loss_func = self.select_loss_func(self.loss_func_type)

        # initialize optimizer
        self.optimizer_type = optimizer_type if optimizer_type else "adam"
        self.learning_rate = learning_rate
        if self.optimizer_type == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == "SGD":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        # start training
        self.train_loss_history = []
        for epoch in range(max_epoch):

            print(f"Epoch {epoch+1}\n-------------------------------")

            batch_loss = []
            batch_trained_sample = 0
            for index_batch, sample_batch in enumerate(train_dataloader):

                # query inputs - reshape and move to gpu
                query_input_ids = sample_batch.query.input_ids.squeeze(1).to(self.device)
                query_attn_mask = sample_batch.query.attn_mask.squeeze(1).to(self.device)
                # evidence inputs - reshape and move to gpu
                evid_input_ids = torch.flatten(sample_batch.evid.input_ids, 0, 1).to(self.device)
                evid_attn_mask = torch.flatten(sample_batch.evid.attn_mask, 0, 1).to(self.device)
                # positive evidence position info
                is_positive = sample_batch.is_positive

                # forward pass the input through the biencoder model 
                query_vector, evid_vector = self.model(query_ids=query_input_ids,
                                                       query_attn_mask=query_attn_mask,
                                                       evid_ids=evid_input_ids,
                                                       evid_attn_mask=evid_attn_mask)

                query_vector = query_vector.to(self.device)
                evid_vector = evid_vector.to(self.device)
                
                # calculate the loss
                loss = loss_func(query_vector=query_vector,
                                 evidence_vector=evid_vector,
                                 is_positive=is_positive)

                # backpropadation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_trained_sample += len(query_input_ids)
                # print info and store history
                if index_batch % 10 == 0 or batch_trained_sample == size:
                    print(f"loss: {loss:>7f}  [{batch_trained_sample:>5d}/{size:>5d}]")
                    batch_loss.append(float(loss))
                
                del query_input_ids, query_attn_mask, query_vector, evid_input_ids, evid_attn_mask, evid_vector 

            self.train_loss_history.append(batch_loss)
            print()
        
        self.model.eval()
        self.model.encoder_0.eval()
        if self.model.encoder_1 is not None:
            self.model.encoder_1.eval()
        
        print("Training Done!")
        return self.train_loss_history


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
        # in-batch negative sampling is done by 
        # taking thte dot product of query_vec and the transpose of evidence_vec
        return torch.mm(query_vec, evidence_vec.transpose(dim0=0, dim1=1)) / 100.0


    def cosine_similarity(self, query_vec: T, evidence_vec: T):
        # cosine similarity of two vectors is the dot product of two normalised vector
        normalized_query_vec = F.normalize(query_vec, p=2, dim=1)
        normalized_evidence_vec = F.normalize(evidence_vec, p=2, dim=1)

        return torch.mm(normalized_query_vec, normalized_evidence_vec.transpose(dim0=0, dim1=1)) * 10.0


    def negative_likelihood_loss(self, query_vector: T, evidence_vector: T, is_positive: T):        
        similarity_func = self.select_similarity_func(self.model.similarity_func_type)

        # calculate similarity score
        similarity_score = similarity_func(query_vec=query_vector, evidence_vec=evidence_vector)
        
        positive_mask = self.create_positive_mask(is_positive=is_positive,
                                                  shape=similarity_score.shape).to(self.device)
        # compute log softmax score
        log_softmax_score = - F.log_softmax(similarity_score, dim=-1).to(self.device)
        
        positive_log_softmax_score = torch.sum(log_softmax_score * positive_mask, dim=-1)
        
        # return negative log likelihood of the positive evidences
        return torch.mean(positive_log_softmax_score).to(self.device)

    
    def create_positive_mask(self, is_positive, shape):
        mask = torch.zeros(shape)
        
        for idx, positive_end in enumerate(is_positive):
            start_idx = (self.train_dataset.evidence_num  * idx) + idx
            mask[idx, start_idx: start_idx + positive_end] = 1
            
        return mask

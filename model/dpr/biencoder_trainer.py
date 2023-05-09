import torch
from torch import Tensor as T
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .biencoder import BiEncoder
from .biencoder_dataset import BiEncoderDataset
from .evidence_dataset import EvidenceDataset


class BiEncoderTrainer():
    def __init__(self, 
                 model:BiEncoder=None,
                 train_data: BiEncoderDataset=None,
                 validate_data : BiEncoderDataset=None,
                 evidence_data: EvidenceDataset=None,
                 shuffle: bool=True,
                 batch_size: int=64) -> None:
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        
        self.model = model.to(self.device)
        self.train_data = train_data
        self.validate_data = validate_data
        self.evidence_data = evidence_data
        
        self.shuffle =shuffle
        self.batch_size = batch_size
        
        if train_data is not None:
            self.set_dataloader()
    
    def set_dataloader(self) -> None:
        self.train_dataloader = DataLoader(self.train_data,
                                           batch_size=self.batch_size,
                                           shuffle=self.shuffle, 
                                           num_workers=2, 
                                           collate_fn=self.train_data.train_collate_fn)
        
        self.train_evaluate_dataloader = DataLoader(self.train_data,
                                                    batch_size=self.batch_size, 
                                                    shuffle=False,
                                                    collate_fn=self.train_data.evaluate_collate_fn)
        
        self.dev_evaluate_dataloader = DataLoader(self.validate_data,
                                                  batch_size=self.batch_size,
                                                  shuffle=False, 
                                                  collate_fn=self.validate_data.evaluate_collate_fn)
        
        self.evidence_dataloader = DataLoader(self.evidence_data, 
                                              batch_size=1000,
                                              drop_last=False,
                                              shuffle=False) 
            
            
    def train(self, 
              train_data: BiEncoderDataset=None,
              validate_data : BiEncoderDataset=None,
              evidence_data: EvidenceDataset=None,           
              shuffle: bool=True,
              max_epoch: int=10, 
              loss_func_type: str=None,
              similarity_func_type : str=None,
              optimizer_type: str=None, 
              learning_rate: float=0.001):

        # setup dataloader
        self.shuffle = shuffle
        self.train_data = train_data if train_data else self.train_data
        self.validate_data = validate_data if validate_data else self.validate_data
        self.evidence_data = evidence_data if evidence_data else self.evidence_data
        
        if train_data is not None or validate_data is not None or evidence_data is not None:
            self.set_dataloader()
        
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
            trained_sample = 0
            for index_batch, sample_batch in enumerate(self.train_dataloader):

                query = sample_batch.query
                evid = sample_batch.evid
                is_positive = sample_batch.is_positive

                # forward pass the input through the biencoder model 
                query_vector, evid_vector = self.model(query_ids=query.input_ids.to(self.device),
                                                       query_segment=query.segments.to(self.device),
                                                       query_attn_mask=query.attn_mask.to(self.device),
                                                       evid_ids=evid.input_ids.to(self.device),
                                                       evid_segment=evid.segments.to(self.device),
                                                       evid_attn_mask=evid.attn_mask.to(self.device))

                # calculate the loss
                loss = loss_func(query_vector=query_vector.to("cpu"),
                                 evidence_vector=evid_vector.to("cpu"),
                                 is_positive=is_positive)

                # backpropadation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print info and store history
                if index_batch % 10 == 0:
                    trained_sample += len(query.input_ids)
                    print(f"loss: {loss:>7f}  [{trained_sample:>5d}/{size:>5d}]")
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


    def cosine_similarity(self, query_vec: T, evidence_vec: T):
        # cosine similarity of two vectors is the dot product of two normalised vector
        normalized_query_vec = F.normalize(query_vec, p=2, dim=1)
        normalized_evidence_vec = F.normalize(evidence_vec, p=2, dim=1)

        return self.dot_similarity(query_vec=normalized_query_vec,
                                   evidence_vec=normalized_evidence_vec)


    def negative_likelihood_loss(self, query_vector: T, evidence_vector: T, is_positive: T):        
        similarity_func = self.select_similarity_func(self.similarity_func_type)

        # ensure the vectors only has 2 dim
        if len(query_vector.shape) == 3:
            query_vector = query_vector.squeeze(1)

        if len(evidence_vector.shape) == 3:
            evidence_vector = torch.flatten(evidence_vector, 0, 1)

        # calculate similarity score and scale down the value for the sake of performance
        similarity_score = similarity_func(query_vec=query_vector, evidence_vec=evidence_vector) / 10.0
        
        positive_mask = self.create_positive_mask(is_positive=is_positive, shape=similarity_score.shape)
        
        log_softmax_score = F.log_softmax(similarity_score, dim=1)
        
        return - torch.mean(log_softmax_score * positive_mask).to(self.device)

    
    def create_positive_mask(self, is_positive, shape):
        
        mask = torch.zeros(shape)
        
        for idx, positive_end in enumerate(is_positive):
            mask[idx, idx: idx+positive_end] = 1
            
        return mask
    
    
    def get_evidence_embbed(self):
        evidence_embbed = []
        evidence_tag = []
        
        for batch_sample in tqdm(self.evidence_dataloader):
            evidence = batch_sample.evidence
            evid_input_ids = evidence.input_ids.to(self.device)
            evid_segments = evidence.segments.to(self.device)
            evid_attent_mask = evidence.attn_mask.to(self.device)
            
            evid_embbed = self.model.encode_evidence(input_ids=evid_input_ids,
                                                     segments=evid_segments,
                                                     attent_mask=evid_attent_mask)
            
            evid_embbed_cpu = evid_embbed.cpu()
            
            del evid_embbed, evid_input_ids, evid_segments, evid_attent_mask
            
            evidence_embbed.append(evid_embbed_cpu)
            evidence_tag.extend(batch_sample)
        
        return torch.cat(evidence_embbed), evidence_tag

    
    
    def evaluate(self, dataloader: DataLoader, evid_embbed_set):
        
        f_score = []
        evid_embbed, evid_tag = evid_embbed_set
        
        for batch_sample in tqdm(dataloader):
            query = batch_sample.query
            evid_tag = batch_sample.evid_tag
            
            query_embbed = self.model.encode_query(input_ids=query.input_ids,
                                                  segments=query.segments,
                                                  attent_mask=query.attn_mask)
            
            similarity_score = torch.matmul(query_embbed, evid_embbed.transpose(dim0=0, dim1=1))
            top_k = torch.topk(similarity_score, k=5, dim=1).indices

            for query_idx, tags in enumerate(evid_tag):
                evidence_correct = 0
                
                pred_evidences = [evid_tag[evid_idx] for evid_idx in top_k[query_idx]]
                
                for tag in tags:
                    if tag in pred_evidences:
                        evidence_correct += 1
                if evidence_correct > 0:
                    evid_recall = float(evidence_correct) / len(tags)
                    evid_precision = float(evidence_correct) / len(pred_evidences)
                    evid_f_score = (2 * evid_precision * evid_recall) / (evid_precision + evid_recall)
                else:
                    evid_f_score = 0
                    
                f_score.append(evid_f_score)

        mean_f_score = (f_score) / len(f_score)
        print(f"Biencoder F-score: {mean_f_score:>6f}")

        return mean_f_score
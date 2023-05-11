from typing import Tuple
import json
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor as T
from torch.utils.data import DataLoader
from .evidence_dataset import EvidenceDataset
from .biencoder_dataset import BiEncoderDataset


class BiEncoder(nn.Module):
    def __init__(self, query_model: nn.Module, evid_model: nn.Module=None) -> None:
        super(BiEncoder, self).__init__()
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.encoder_0 = query_model.to(self.device)
        self.encoder_1 = evid_model.to(self.device) if evid_model else None
        
        # freeze encoder layer param
        modules_0 = [self.encoder_0.embeddings, *self.encoder_0.encoder.layer[:1]]
        for module in modules_0:
            for param in module.parameters():
                param.requires_grad = False
                
        if self.encoder_1:
            modules_1 = [self.encoder_1.embeddings, *self.encoder_1.encoder.layer[:1]]
            for module in modules_1:
                for param in module.parameters():
                    param.requires_grad = False 


    def forward(self, query_ids, query_segment, query_attn_mask, evid_ids, evid_segment, evid_attn_mask) -> Tuple[T, T]:
        
        _query_seq, query_pooler_out, _query_hidden = self.get_representation(sub_model=self.encoder_0,
                                                                              ids=query_ids,
                                                                              segments=query_segment,
                                                                              attent_mask=query_attn_mask)
        
        if self.encoder_1:
            _evid_seq, evid_pooler_out, _evid_hidden = self.get_representation(sub_model=self.encoder_1,
                                                                               ids=evid_ids,
                                                                               segments=evid_segment,
                                                                               attent_mask=evid_attn_mask)
        else:
            _evid_seq, evid_pooler_out, _evid_hidden = self.get_representation(sub_model=self.encoder_0,
                                                                               ids=evid_ids,
                                                                               segments=evid_segment,
                                                                               attent_mask=evid_attn_mask)
        
        return query_pooler_out, evid_pooler_out


    def get_representation(self, sub_model: nn.Module=None, ids: T=None, segments: T=None, attent_mask: T=None) -> Tuple[T, T, T]:
        # make sure the model is add_pooling_layer = True, return_dict=True
        
        if sub_model.training:
            out = sub_model(input_ids=ids,
                            token_type_ids=segments,
                            attention_mask=attent_mask)

        else:
            with torch.no_grad():
                out = sub_model(input_ids=ids,
                                token_type_ids=segments,
                                attention_mask=attent_mask)
        
        sequence = out.last_hidden_state
        pooler_output = out.pooler_output
        hidden_state = out.hidden_states
        
        return sequence, pooler_output, hidden_state
    
    
    def encode_query(self, input_ids: T, segments: T, attent_mask: T):
            
        self.encoder_0.eval()
        with torch.no_grad():
            out = self.encoder_0(input_ids=input_ids,
                                 token_type_ids=segments,
                                 attention_mask=attent_mask)
        
        query_pooler_output = out.pooler_output
        
        self.encoder_0.train()

        return query_pooler_output


    def encode_evidence(self, input_ids: T, segments: T, attent_mask: T):
        
        if self.encoder_1 is not None:

            self.encoder_1.eval()
            with torch.no_grad():
                out = self.encoder_1(input_ids=input_ids,
                                     token_type_ids=segments,
                                     attention_mask=attent_mask)
            
            evid_pooler_output = out.pooler_output
            self.encoder_1.train()
        else:
            evid_pooler_output = self.encode_query(input_ids=input_ids,
                                                   segments=segments,
                                                   attent_mask=attent_mask)

        return evid_pooler_output
    
    
    def get_evidence_embed(self, evidence_dataset: EvidenceDataset=None, batch_size: int=64, output_file_path: str=None):
        evidence_embed = []
        evidence_tag = []
        
        evidence_dataloader = DataLoader(evidence_dataset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         collate_fn=evidence_dataset.tok_evidence_collate_fn)
        with torch.no_grad():
            for batch_sample in tqdm(evidence_dataloader):
                evidence = batch_sample.evidence_tok
                
                evid_input_ids = evidence.input_ids.squeeze(1).to(self.device)
                evid_segments = evidence.segments.squeeze(1).to(self.device)
                evid_attent_mask = evidence.attn_mask.squeeze(1).to(self.device)
                
                evid_embed = self.encode_evidence(input_ids=evid_input_ids,
                                                  segments=evid_segments,
                                                  attent_mask=evid_attent_mask)
                
                evid_embed_cpu = F.normalize(evid_embed, p=2, dim=-1).cpu()
                
                del evid_embed, evid_input_ids, evid_segments, evid_attent_mask
                
                evidence_embed.append(evid_embed_cpu)
                evidence_tag.extend(batch_sample.tag)
            
        evidence_embed = torch.cat(evidence_embed)
        
        evid_embed_dict = {tag: embed.tolist() for tag, embed in zip(evidence_tag, evidence_embed)}
        
        if output_file_path is not None:
            print("Exporting...")
            f_out = open("data/output/embed-evidence.json", 'w')
            json.dump(evid_embed_dict, f_out)
            f_out.close()
            print('\033[1A', end='\x1b[2K')
            print("File Exported.")
        
        evid_embed_dict = {"tag": evidence_tag, "evidence": evidence_embed}
        
        print("Embedding Done!")
        return evid_embed_dict


    def retrieve(self, claim_dataset: BiEncoderDataset, embed_evid_data: dict, batch_size: int=64, k: int=5, predict_output_path: str=None):
        
        claim_dataloader = DataLoader(claim_dataset, 
                                      batch_size=batch_size,
                                      shuffle=False,
                                      num_workers=2,
                                      collate_fn=claim_dataset.predict_collate_fn)
        
        evid_embed = torch.tensor(embed_evid_data["evidence"]).transpose(dim0=0, dim1=1).to(self.device)
        tags = embed_evid_data["tag"]
        
        predictions = {}
        
        for batch_claim_sample in tqdm(claim_dataloader):
            
            query = batch_claim_sample.query
            query_tags = batch_claim_sample.query_tag
            
            query_input_ids = query.input_ids.to(self.device)
            query_segments = query.segments.to(self.device)
            query_attn_mask = query.attn_mask.to(self.device)
            
            with torch.no_grad():
                query_embed = self.encode_query(input_ids=query_input_ids,
                                                segments=query_segments,
                                                attent_mask=query_attn_mask)
                
                similarity_score = torch.matmul(F.normalize(query_embed.flatten(1), p=2, dim=-1), evid_embed)

                top_ks = torch.topk(similarity_score, k=k, dim=1).indices
            
                for query_tag, top_k in zip(query_tags, top_ks):
                    predictions[query_tag] = [tags[idx] for idx in top_k]
                
            del query_input_ids, query_segments, query_attn_mask, query_embed, similarity_score, top_ks
        
        del evid_embed
        
        if predict_output_path is not None:
            print("Exporting...")
            f_out = open(predict_output_path, 'w')
            json.dump(predictions, f_out)
            f_out.close()
            print('\033[1A', end='\x1b[2K')
            print("File Exported.")
        
        print("Retrieve Done!")
        return (predictions)
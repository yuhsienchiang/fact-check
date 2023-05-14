import json
import pandas as pd

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from collections import namedtuple

EvidenceDataSample = namedtuple('EvidenceDataSample', ["tag", "evidence"])
EvidenceDataPassage = namedtuple('EvidenceDataPassage', ['input_ids', 'segments', 'attn_mask'])
EvidenceEmbedDataSample = namedtuple('EvidenceEmbedDataSample', ["tag", "evidence_embed"])
EvidenceTokDataSample = namedtuple('EvidenceTokDataSample', ["tag", "evidence_tok"])

class EvidenceDataset(Dataset):
    def __init__(self,
                 evidence_file_path: str,
                 tokenizer: BertTokenizer=None,
                 lower_case :bool=False,
                 max_padding_length: int=12,
                 rand_seed: int=None) -> None:
        
        super(EvidenceDataset, self).__init__()
        
        self.evidence_file_path = evidence_file_path

        self.tokenizer = tokenizer if tokenizer else BertTokenizer.from_pretrained("bert-base-uncased")
        
        self.lower_case = lower_case
        self.max_padding_length = max_padding_length
        self.rand_seed = rand_seed

        self.raw_evidence_data = None

        self.evidence_data = None

        self.load_data()


    def __len__(self) -> int:
        return len(self.raw_evidence_data)


    def __getitem__(self, idx) -> EvidenceDataSample:
        
        data = self.evidence_data.iloc[idx]
        
        return EvidenceDataSample(tag=data.tag,
                                  evidence=data.evidence)


    def load_data(self) -> None:
        self.raw_evidence_data = json.load(open(self.evidence_file_path))
        
        normalized_evidence_data = [{"tag": key, 
                                     "evidence": value
                                    } 
                                    for (key, value) in self.raw_evidence_data.items()]

        self.evidence_data = pd.json_normalize(normalized_evidence_data)


    def clean_text(self, context: str, lower_case: bool=False) -> str:
        return context.lower() if lower_case else context

    
    def tok_evidence_collate_fn(self, batch):
        batch_evid_input_ids = []
        batch_evid_segments = []
        batch_evid_attn_mask = []
        batch_evid_tag = []
        
        for batch_sample in batch:
            
            evidence_text = self.clean_text(batch_sample.evidence, lower_case=self.lower_case)
            evidence_tag = batch_sample.tag
            
            evidence_encoding = self.tokenizer(text=evidence_text,
                                               add_special_tokens=True,
                                               padding='max_length',
                                               truncation='longest_first',
                                               max_length=self.max_padding_length,
                                               return_tensors='pt')
            
            batch_evid_input_ids.append(evidence_encoding.input_ids)
            batch_evid_segments.append(evidence_encoding.token_type_ids)
            batch_evid_attn_mask.append(evidence_encoding.attention_mask)
            batch_evid_tag.append(evidence_tag)
        
        evidence_passage = EvidenceDataPassage(input_ids=torch.stack(batch_evid_input_ids, dim=0),
                                               segments=torch.stack(batch_evid_segments, dim=0),
                                               attn_mask=torch.stack(batch_evid_attn_mask, dim=0))
         
        return EvidenceTokDataSample(tag=batch_evid_tag,
                                     evidence_tok=evidence_passage)
        
    def embed_evidence_collate_fn(self, batch):
        batch_evid_embed = []
        batch_evid_tag = []
        
        for batch_sample in batch:
            
            batch_evid_embed.appen(batch_sample.evidence)
            batch_evid_tag.append(batch_sample.tag)
        
        return EvidenceEmbedDataSample(tag=batch_evid_tag,
                                       evidence_embed=torch.stack(batch_evid_embed, dim=0))
import json
import pandas as pd

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from collections import namedtuple

EvidenceDataSample = namedtuple('EvidenceDataSample', ["tag", "evidence"])
EvidenceDataPassage = namedtuple('EvidenceDataPassage', ['input_ids', 'segments', 'attn_mask'])


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
        
        evidence_text = self.clean_text(data["evidence"], lower_case=self.lower_case)
        evidence_encoding = self.tokenizer(text=evidence_text,
                                           add_special_tokens=True,
                                           padding='max_length',
                                           truncation='longest_first',
                                           max_length=self.max_padding_length,
                                           return_tensors='pt')
        
        evidence_passage = EvidenceDataPassage(input_ids=evidence_encoding.input_ids,
                                               segments=evidence_encoding.token_type_ids,
                                               attn_mask=evidence_encoding.attention_mask)
        
        return EvidenceDataSample(tag=data.tag,
                                  evidence=evidence_passage)


    def load_data(self) -> None:
        self.raw_evidence_data = json.load(open(self.evidence_file_path))
        
        normalized_evidence_data = [{"tag": key, 
                                     "evidence": value
                                    } 
                                    for (key, value) in self.raw_evidence_data.items()]

        self.evidence_data = pd.json_normalize(normalized_evidence_data)


    def clean_text(self, context: str, lower_case: bool=False) -> str:
        
        context = context.replace("`", "'")
        context = context.replace(" 's", "'s")
        
        return context.lower() if lower_case else context

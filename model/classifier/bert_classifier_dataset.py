import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from collections import namedtuple

BertClassifierSample = namedtuple('BertClassifierSample', ['query_tag', 'query_text', 'query_label', 'evid_tag', 'evid_text'])
BertClassifierTrainSample = namedtuple('BertClassifierTrainSample', ['query_tag', 'query_label', 'text_sequence'])
BertClassifierPredictSample = namedtuple('BertClassifierPredictSample', ['query_tag', 'text_sequence'])
BertClassifierPassage = namedtuple('BertClassifierPassage', ['input_ids', 'segments', 'attn_mask'])

CLASS_TO_IDX = {"SUPPORTS": 0, "REFUTES": 1, "NOT_ENOUGH_INFO": 2, "DISPUTED": 3}
IDX_TO_CLASS = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT_ENOUGH_INFO", 3: "DISPUTED"}


class BertClassifierDataset(Dataset):
    def __init__(self,
                 claim_file_path: str,
                 data_type: str,
                 evidence_file_path: str=None,
                 predict_evidence_file_path: str=None,
                 tokenizer: BertTokenizer=None,
                 lower_case: bool=False,
                 max_padding_length: int=12,
                 rand_seed: int=None) -> None:
        
        super(BertClassifierDataset, self).__init__()
        
        self.claim_file_path = claim_file_path
        self.evidence_file_path = evidence_file_path
        self.predict_evidence_file_path = predict_evidence_file_path
        
        self.predict = True if data_type == "predict" else False
        
        self.tokenizer = tokenizer if tokenizer else BertTokenizer.from_pretrained("bert-base-uncased")
        
        self.lower_case = lower_case
        self.max_padding_length = max_padding_length
        self.rand_seed = rand_seed
        
        self.raw_claim_data = None
        self.raw_evidence_data = None
        
        self.claim_data = None
        self.evidences_data = None
        
        self.load_data()
        
        
    def __len__(self) -> int:
        return len(self.raw_claim_data)
    
    
    def __getitem__(self, idx):
        
        data = self.claim_data.iloc[idx]
        
        query_tag = data["tag"]
        query_text = self.clean_text(data["claim_text"])
        query_label = CLASS_TO_IDX[data["claim_label"]] if self.predict is False else None
        
        evidence_tag = data["evidence_tag"]
        evidence_text = [self.clean_text(evid, lower_case=self.lower_case) 
                         for evid in map(self.raw_evidence_data.get, evidence_tag)]
        
        return BertClassifierSample(query_tag=query_tag,
                                    query_label=query_label,
                                    query_text=query_text,
                                    evid_tag=evidence_tag,
                                    evid_text=evidence_text)
    
    
    def load_data(self) -> None:
        
        self.raw_claim_data = json.load(open(self.claim_file_path))
        self.raw_evidence_data = json.load(open(self.evidence_file_path))
        
        if self.predict is False:
            normalized_claim_data = [{"tag": key,
                                      "claim_text": value["claim_text"],
                                      "claim_label": value["claim_label"],
                                      "evidence_tag": value["evidences"]
                                      } 
                                     for (key, value) in self.raw_claim_data.items()] 
        else:
            self.raw_predice_evidence_data = json.load(open(self.predict_evidence_file_path))
            
            normalized_claim_data = [{"tag": key,
                                     "claim_text": value["claim_text"],
                                     "evidence_tag": self.raw_predice_evidence_data[key]
                                     }
                                     for (key, value) in self.raw_claim_data.items()]

        normalized_evidence_data = [{"tag": key, 
                                     "evidences": value
                                    } 
                                    for (key, value) in self.raw_evidence_data.items()]
        
        self.claim_data = pd.json_normalize(normalized_claim_data)
        self.evidences_data = pd.json_normalize(normalized_evidence_data)
        
        
    def clean_text(self, context: str, lower_case: bool=False) -> str:
        
        context = context.replace("`", "'")
        context = context.replace(" 's", "'s")
        
        return context.lower() if lower_case else context
    
    
    def train_collate_fn(self, batch):
        batch_text_sequence_input_ids = []
        batch_text_sequence_segments = []
        batch_text_sequence_attn_mask = [] 
        batch_label = []
        batch_tag = []
        
        for batch_sample in batch:
            claim_text = batch_sample.query_text
            evidence_text = batch_sample.evid_text
            label = batch_sample.query_label
            tag = batch_sample.query_tag
        
            input_sequence = self.tokenizer.sep_token.join([claim_text] + evidence_text)
        
            encoding = self.tokenizer(text=input_sequence,
                                  add_special_tokens=True,
                                  padding="max_length",
                                  truncation="longest_first",
                                  max_length=self.max_padding_length,
                                  return_tensors="pt")
        
            batch_text_sequence_input_ids.append(encoding.input_ids)
            batch_text_sequence_segments.append(encoding.token_type_ids) 
            batch_text_sequence_attn_mask.append(encoding.attention_mask) 
            batch_label.append(label)
            batch_tag.append(tag)
            
            text_sequences_passage = BertClassifierPassage(input_ids=torch.stack(batch_text_sequence_input_ids, dim=0),
                                                           segments=torch.stack(batch_text_sequence_segments, dim=0),
                                                           attn_mask=torch.stack(batch_text_sequence_attn_mask, dim=0))
        
        return BertClassifierTrainSample(text_sequence=text_sequences_passage,
                                         query_label=torch.tensor(batch_label),
                                         query_tag=batch_tag)
    
    
    def evaluate_collate_fn(self, batch): 
        batch_text_sequence_input_ids = []
        batch_text_sequence_segments = []
        batch_text_sequence_attn_mask = [] 
        batch_tag = []
        
        for batch_sample in batch:
            claim_text = batch_sample.query_text
            evidence_text = batch_sample.evid_text
            tag = batch_sample.query_tag
        
            input_sequence = self.tokenizer.sep_token.join([claim_text] + evidence_text)
        
            encoding = self.tokenizer(text=input_sequence,
                                  add_special_tokens=True,
                                  padding="max_length",
                                  truncation="longest_first",
                                  max_length=self.max_padding_length,
                                  return_tensors="pt")
        
            batch_text_sequence_input_ids.append(encoding.input_ids)
            batch_text_sequence_segments.append(encoding.token_type_ids) 
            batch_text_sequence_attn_mask.append(encoding.attention_mask) 
            batch_tag.append(tag)
            
            text_sequences_passage = BertClassifierPassage(input_ids=torch.stack(batch_text_sequence_input_ids, dim=0),
                                                           segments=torch.stack(batch_text_sequence_segments, dim=0),
                                                           attn_mask=torch.stack(batch_text_sequence_attn_mask, dim=0))
        
        return BertClassifierPredictSample(text_sequence=text_sequences_passage,
                                           query_tag=batch_tag)
    
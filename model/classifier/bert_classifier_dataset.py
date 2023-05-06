import json
import pandas as pd
from torch.utils.data import Dataset
from transformers import BertTokenizer
from collections import namedtuple

BertClassifierPassage = namedtuple('BiEncoderPassage', ['input_ids', 'segments', 'attn_mask'])

CLASS_TO_IDX = {"SUPPORTS": 0, "REFUTES": 1, "NOT_ENOUGH_INFO": 2, "DISPUTED": 3}
IDX_TO_CLASS = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT_ENOUGH_INFO", 3: "DISPUTED"}


class BertClassifierDataset(Dataset):
    def __init__(self,
                 claim_file_path: str,
                 evidence_file_path: str=None,
                 tokenizer: BertTokenizer=None,
                 max_padding_length: int=12,
                 rand_seed: int=None) -> None:
        
        super(BertClassifierDataset, self).__init__()
        
        self.claim_file_path = claim_file_path
        self.evidence_file_path = evidence_file_path
        
        self.tokenizer = tokenizer if tokenizer else BertTokenizer.from_pretrained("bert-base-uncased")
        
        self.max_padding_length = max_padding_length
        self.rand_seed = rand_seed
        
        self.raw_claim_data = None
        self.raw_evidence_data = None
        
        self.claim_data = None
        self.evidences_data = None
        
        self.load_data()
        
        
    def __len__(self):
        return len(self.raw_claim_data)
    
    
    def __getitem__(self, idx):
        
        data = self.claim_data.iloc[idx]
        
        claim_text = self.clean_text(data["claim_text"])
        evidence_text = self.clean_text(data["evidence"])
        label = CLASS_TO_IDX[data["claim_label"]]
        
        encoding = self.tokenizer(text=claim_text,
                                  text_pair=evidence_text,
                                  add_special_tokens=True,
                                  padding="max_length",
                                  truncation="longest_first",
                                  max_length=self.max_padding_length,
                                  return_tensors="pt")
        
        text_pair_passage = BertClassifierPassage(input_ids=encoding.input_ids,
                                                  segments=encoding.token_type_ids,
                                                  attn_mask=encoding.attention_mask)
        
        return text_pair_passage, label
    
    
    def load_data(self) -> None:
        
        self.raw_claim_data = json.load(open(self.claim_file_path))
        self.raw_evidence_data = json.load(open(self.evidence_file_path))
        
        normalized_claim_data = [{"tag": key,
                                  "claim_text": value["claim_text"],
                                  "claim_label": value["claim_label"],
                                  "evidence": evid
                                  } 
                                 for (key, value) in self.raw_claim_data.items()
                                 for evid in list(map(self.raw_evidence_data.get, value["evidences"]))] 
        
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
    
 
        
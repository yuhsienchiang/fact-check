import json
import pandas as pd
from torch.utils.data import Dataset
from transformers import BertTokenizer
from typing import List

 # TO-Do
 # create method to process text: remove some symbols   

class BiEncoderDataset(Dataset):
    def __init__(self,
                 claim_file_path: str,
                 evidence_file_path: str=None,
                 tokenizer: BertTokenizer=None,
                 max_padding_length: int=12,
                 neg_evidence_num: int=2,
                 rand_seed: int=None) -> None:
        
        self.claim_file_path = claim_file_path
        self.evidence_file_path = evidence_file_path
        
        self.tokenizer = tokenizer if tokenizer else BertTokenizer.from_pretrained("bert-base-uncased")
        
        self.max_padding_length = max_padding_length
        self.neg_evidence_num = neg_evidence_num
        self.rand_seed = rand_seed
        
        self.raw_claim_data = None
        self.raw_evidence_data = None
        
        self.id_labels = None
        self.querys = None
        self.evidences = None

    def __len__(self):
        return (len(self.raw_data))

    def __getitem__(self, idx):
        data = self.claim_data.iloc[idx]
        sample = BiencoderSample()
        
        query = self.tokenizer.encode(text=data["claim_text"],
                                      add_special_tokens=True,
                                      padding='max_length',
                                      max_length=self.max_padding_length,
                                      return_tensors='pt')
        
        posivite_evidence = [self.tokenizer.encode(text=evidence,
                                                   add_special_tokens=True,
                                                   padding='max_length',
                                                   max_length=self.max_padding_length,
                                                   return_tensors='pt')
                             for evidence in data["evidences"]]
        
        negtive_evidence = self.evidences.sample(n=self.neg_evidence_num, random_state=self.rand_seed)["evidences"].tolist()
        
    
        sample.query = query
        sample.positive_evidence = posivite_evidence
        sample.negative_evidence = negtive_evidence    
        
        return sample
    
    
    def load_data(self) -> None:
        self.raw_claim_data = json.load(open(self.claim_file_path))
        self.raw_evidence_data = json.load(open(self.evidence_file_path))
        
        
        normalized_claim_data = [{"tag": key, 
                                  "claim_text": value["claim_text"],
                                  "claim_label": value["claim_label"],
                                  "evidences": list(map(self.raw_evidence_data.get, value["evidences"]))
                                 }
                                 for (key, value) in self.raw_claim_data.items()]
        
        normalized_evidence_data = [{"tag": key, 
                                     "evidences": value
                                    } 
                                    for (key, value) in self.raw_evidence_data.items()]
        
        self.claim_data = pd.json_normalize(normalized_claim_data)
        self.evidences_data = pd.json_normalize(normalized_evidence_data)
        
        
class BiencoderSample(object):
    def __init__(self, 
                 query: str=None,
                 positive_evidences: List[str]=None,
                 negative_evidence: List[str]=None) -> None:
        
        self.query = query
        self.positive_evidence = positive_evidences
        self.negative_evidence = negative_evidence
    
    
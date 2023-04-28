import json
import pandas as pd
import torch
from torch import Tensor as T
from torch.utils.data import Dataset
from transformers import BertTokenizer
from collections import namedtuple

 # TO-Do
 # create method to process text: remove some symbols   
BiEncoderSample = namedtuple('BiEncoderSample', ['query', 'evid'])
BiEncoderPassage = namedtuple('BiEncoderPassage', ['input_ids', 'segments', 'attn_mask', 'posit_neg_idx'])
PADDING_TENSOR_ELEMENT = -1

class BiEncoderDataset(Dataset):
    def __init__(self,
                 claim_file_path: str,
                 evidence_file_path: str=None,
                 tokenizer: BertTokenizer=None,
                 max_padding_length: int=12,
                 neg_evidence_num: int=2,
                 rand_seed: int=None) -> None:
        
        super(BiEncoderDataset, self).__init__() 
        
        self.claim_file_path = claim_file_path
        self.evidence_file_path = evidence_file_path
        
        self.tokenizer = tokenizer if tokenizer else BertTokenizer.from_pretrained("bert-base-uncased")
        
        self.max_padding_length = max_padding_length
        self.neg_evidence_num = neg_evidence_num
        self.rand_seed = rand_seed
        
        self.raw_claim_data = None
        self.raw_evidence_data = None
        
        self.claim_data = None
        self.evidences_data = None
        
        self.load_data()


    def __len__(self):
        return (len(self.raw_claim_data))


    def __getitem__(self, idx):
        # fetch sample
        data = self.claim_data.iloc[idx]
        
        # clean up claim text
        claim_text = self.clean_text(data["claim_text"])
        
        # tokenized claim text
        query_tokenized = self.tokenizer(text=claim_text,
                                         add_special_tokens=True,
                                         padding='max_length',
                                         truncation="longest_first",
                                         max_length=self.max_padding_length,
                                         return_tensors='pt')
        # store claim tokens in namedtuple format
        query_passage = BiEncoderPassage(query_tokenized.input_ids, 
                                         query_tokenized.token_type_ids,
                                         query_tokenized.attention_mask,
                                         posit_neg_idx=torch.tensor([-1,-1]))
        
        # prepare positive evid text
        positive_evid_textset = [self.clean_text(evidence) for evidence in data["evidences"]]
        positive_evid_end_idx = len(positive_evid_textset)
        positive_evid_textset = positive_evid_textset + ["[PAD]"] * (self.max_evidence_num - positive_evid_end_idx)
        negative_evid_start_idx = len(positive_evid_textset)
        # prepare negative evid text
        negative_evid_sample = self.evidences_data.sample(n=self.neg_evidence_num, random_state=self.rand_seed)["evidences"].tolist()
        negative_evid_textset = [self.clean_text(neg_evidence) for neg_evidence in negative_evid_sample]
        # combine positive and negative evid into a single textset
        evid_textset = positive_evid_textset + negative_evid_textset
        
        # tokenized evidence text
        evid_tokenized = self.tokenizer(text=evid_textset,
                                        add_special_tokens=True,
                                        padding='max_length',
                                        truncation='longest_first',
                                        max_length=self.max_padding_length,
                                        return_tensors='pt')
        # store evidence token in namedtuple format
        evidence_passage = BiEncoderPassage(evid_tokenized.input_ids,
                                            evid_tokenized.token_type_ids,
                                            evid_tokenized.attention_mask,
                                            posit_neg_idx=torch.tensor([positive_evid_end_idx, negative_evid_start_idx]))

        return BiEncoderSample(query_passage, evidence_passage)
    
    
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
        self.max_evidence_num = self.claim_data["evidences"].apply(lambda x: len(x)).max()
        
        
    def clean_text(self, context: str) -> str:
        context = context.replace("`", "'")
        context = context.replace(" 's", "'s")
        
        return context
    
    
    def pad_tensor(self, evidence: T):
        
        input_ids = evidence.input_ids
        segments = evidence.token_type_ids
        attn_mask = evidence.attention_mask
        
        if self.max_evidence_num == input_ids.shape[0]:
            return evidence
        else:
            pad_tensor_num = self.max_evidence_num - input_ids.shape[0] 
            
            pad_tensor = torch.full((pad_tensor_num, self.max_padding_length), PADDING_TENSOR_ELEMENT)
            
            evidence.input_ids = torch.cat((input_ids, pad_tensor), dim=0)
            evidence.token_type_ids = torch.cat((segments, pad_tensor), dim=0)
            evidence.attention_mask = torch.cat((attn_mask, pad_tensor), dim=0)
            
            return evidence
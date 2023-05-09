import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from collections import namedtuple

BiEncoderTrainSample = namedtuple('BiEncoderTrainSample', ['query', 'evid', 'is_positive'])
BiEncoderEvaluateSample = namedtuple('BiEncoderEvaluateSample', ['query', 'evid_tag'])
BiEncoderSample = namedtuple('BiEncoderSample', ['query_text', 'evid_text', 'evid_tag'])
BiEncoderPassage = namedtuple('BiEncoderPassage', ['input_ids', 'segments', 'attn_mask'])


class BiEncoderDataset(Dataset):
    def __init__(self,
                 claim_file_path: str,
                 evidence_file_path: str=None,
                 tokenizer: BertTokenizer=None,
                 lower_case :bool=False,
                 max_padding_length: int=12,
                 evidence_num: int=None,
                 rand_seed: int=None) -> None:

        super(BiEncoderDataset, self).__init__() 

        self.claim_file_path = claim_file_path
        self.evidence_file_path = evidence_file_path

        self.tokenizer = tokenizer if tokenizer else BertTokenizer.from_pretrained("bert-base-uncased")

        self.lower_case = lower_case
        self.max_padding_length = max_padding_length
        self.rand_seed = rand_seed

        self.raw_claim_data = None
        self.raw_evidence_data = None

        self.claim_data = None
        self.evidences_data = None
        
        self.load_data()
        self.max_positive_num = self.claim_data["evidence"].apply(lambda x: len(x)).max()
        
        self.evidence_num = max(evidence_num, self.max_positive_num) if evidence_num is not None else self.max_positive_num


    def __len__(self) -> int:
        return (len(self.raw_claim_data))


    def __getitem__(self, idx):
        # fetch sample
        data = self.claim_data.iloc[idx]
        
        # extract query text and apply cleaning
        query_text = self.clean_text(data["claim_text"], lower_case=self.lower_case)
        # extract evidence_tag
        evidence_tag = data["evidence"]
        # extract evidence text and apply cleaning
        evidence_text = [self.clean_text(evid, lower_case=self.lower_case) for evid in map(self.raw_evidence_data.get, evidence_tag)]
        
        return BiEncoderSample(query_text=query_text,
                               evid_text=evidence_text,
                               evid_tag=evidence_tag)
            
    
    def load_data(self) -> None:
        self.raw_claim_data = json.load(open(self.claim_file_path))
        self.raw_evidence_data = json.load(open(self.evidence_file_path))    

        normalized_claim_data = [{"tag": key,
                                  "claim_text": value["claim_text"],
                                  "claim_label": value["claim_label"],
                                  "evidence": value["evidences"]
                                  } 
                                 for (key, value) in self.raw_claim_data.items()] 

        normalized_evidence_data = [{"tag": key, 
                                     "evidence": value
                                    } 
                                    for (key, value) in self.raw_evidence_data.items()]

        self.claim_data = pd.json_normalize(normalized_claim_data)
        self.evidences_data = pd.json_normalize(normalized_evidence_data)


    def clean_text(self, context: str, lower_case: bool=False) -> str:
        context = context.replace("`", "'")
        context = context.replace(" 's", "'s")
        
        return context.lower() if lower_case else context
    

    def train_collate_fn(self, batch):
        batch_query_input_ids = []
        batch_query_segments = []
        batch_query_attn_mask = []
        batch_evid_input_ids = []
        batch_evid_segments = []
        batch_evid_attn_mask = []
        batch_is_positive = []

        for batch_sample in batch:
            
            query_text = batch_sample.query_text
            evid_text = batch_sample.evid_text

            query_tokenized = self.tokenizer(text=query_text,
                                             add_special_tokens=True,
                                             padding='max_length',
                                             truncation="longest_first",
                                             max_length=self.max_padding_length,
                                             return_tensors='pt')
            
            batch_query_input_ids.append(query_tokenized.input_ids)
            batch_query_segments.append(query_tokenized.token_type_ids)
            batch_query_attn_mask.append(query_tokenized.attention_mask)
            

            # prepare positive evid text
            is_positive = len(evid_text)

            # prepare negative evid text
            # sample (evidence num - positive evid num) negative evidence
            # ensure the total evidence amout for each sample is the same 
            negative_evid_sample = self.evidences_data.sample(n=self.evidence_num-is_positive,
                                                              random_state=self.rand_seed)["evidence"].tolist()
            negative_evid_textset = [self.clean_text(neg_evidence, lower_case=self.lower_case) for neg_evidence in negative_evid_sample]

            # combine positive and negative evid into a single textset
            evid_textset = evid_text + negative_evid_textset

            # tokenized evidence text
            evid_tokenized = self.tokenizer(text=evid_textset,
                                            add_special_tokens=True,
                                            padding='max_length',
                                            truncation='longest_first',
                                            max_length=self.max_padding_length,
                                            return_tensors='pt')
            
            batch_evid_input_ids.append(evid_tokenized.input_ids)
            batch_evid_segments.append(evid_tokenized.token_type_ids)
            batch_evid_attn_mask.append(evid_tokenized.attention_mask)
            batch_is_positive.append(is_positive)
            
        batch_query = BiEncoderPassage(input_ids=torch.stack(batch_query_input_ids, dim=0),
                                       segments=torch.stack(batch_query_segments, dim=0),
                                       attn_mask=torch.stack(batch_query_attn_mask, dim=0))
        
        batch_evid = BiEncoderPassage(input_ids=torch.stack(batch_evid_input_ids, dim=0),
                                      segments=torch.stack(batch_evid_segments, dim=0),
                                      attn_mask=torch.stack(batch_evid_attn_mask, dim=0))
        
        return BiEncoderTrainSample(query=batch_query,
                                    evid=batch_evid,
                                    is_positive=torch.tensor(batch_is_positive))
    
    
    def evaluate_collate_fn(self, batch):
        batch_query_input_ids = []
        batch_query_segments = []
        batch_query_attn_mask = []
        batch_evid_tag = []
        
        for batch_sample in batch:
           query_text = batch_sample.query_text
           evid_tag = batch_sample.evid_tag

           query_tokenized = self.tokenizer(text=query_text,
                                             add_special_tokens=True,
                                             padding='max_length',
                                             truncation="longest_first",
                                             max_length=self.max_padding_length,
                                             return_tensors='pt')

           batch_query_input_ids.append(query_tokenized.input_ids)
           batch_query_segments.append(query_tokenized.token_type_ids)
           batch_query_attn_mask.append(query_tokenized.attention_mask)

           batch_evid_tag.append(evid_tag)

        batch_query = BiEncoderPassage(input_ids=torch.stack(batch_query_input_ids, dim=0),
                                       segments=torch.stack(batch_query_segments, dim=0),
                                       attn_mask=torch.stack(batch_query_attn_mask, dim=0))

        return BiEncoderEvaluateSample(query=batch_query,
                                       evid_tag=batch_evid_tag)
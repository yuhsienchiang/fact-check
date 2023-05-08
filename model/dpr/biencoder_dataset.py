import json
import pandas as pd
from torch.utils.data import Dataset
from transformers import BertTokenizer
from collections import namedtuple

BiEncoderSample = namedtuple('BiEncoderSample', ['query', 'evid', 'is_positive'])
BiEncoderPassage = namedtuple('BiEncoderPassage', ['input_ids', 'segments', 'attn_mask'])
PADDING_TENSOR_ELEMENT = -1


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
        
        self.evidence_num = evidence_num if (evidence_num is not None) and (evidence_num >= self.max_positive_num) else self.max_positive_num  + 3


    def __len__(self) -> int:
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
        query_passage = BiEncoderPassage(input_ids=query_tokenized.input_ids, 
                                         segments=query_tokenized.token_type_ids,
                                         attn_mask=query_tokenized.attention_mask)
        
        # prepare positive evid text
        positive_evid_textset = [self.clean_text(evid, self.lower_case) for evid in data["evidence"]]
        is_positive = len(positive_evid_textset)
        
        # prepare negative evid text
        negative_evid_sample = self.evidences_data.sample(n=self.max_positive_num-is_positive, 
                                                          random_state=self.rand_seed)["evidences"].tolist()
        negative_evid_textset = [self.clean_text(neg_evidence, lower_case=self.lower_case) for neg_evidence in negative_evid_sample]
        
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
        evidence_passage = BiEncoderPassage(input_ids=evid_tokenized.input_ids,
                                            segments=evid_tokenized.token_type_ids,
                                            attn_mask=evid_tokenized.attention_mask)

        return BiEncoderSample(query=query_passage,
                               evid=evidence_passage,
                               is_positive=is_positive)
    
    
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
    
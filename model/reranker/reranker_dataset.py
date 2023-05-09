import json
import pandas as pd
from torch.utils.data import Dataset
import transformers
from transformers import BertTokenizer
from collections import namedtuple

RerankerPassage = namedtuple('RerankerPassage', ['input_ids', 'segments', 'attn_mask'])
RerankerSample = namedtuple('RerankerSample', ['positive_passage', 'negative_passage'])

class RerankerDataset(Dataset):
    def __init__(self,
                 claim_file_path: str,
                 evidence_file_path: str=None,
                 tokenizer: BertTokenizer=None,
                 lower_case: bool=False,
                 max_padding_length: int=12,
                 rand_seed: int=None) -> None:
        
        super(RerankerDataset, self).__init__()
        
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
        
        # suppress undesirable warning message
        transformers.logging.set_verbosity_error() 
        self.load_data()
        
        
    def __len__(self) -> int:
        return len(self.raw_claim_data)
    
    
    def __getitem__(self, idx) -> RerankerSample:
        
        data = self.claim_data.iloc[idx]
        
        claim_text = self.clean_text(data["claim_text"], lower_case=self.lower_case)
        positive_evidence_text = self.clean_text(data["evidence"], lower_case=self.lower_case)
        
        negative_evidenct_sample = self.evidences_data.sample(n=1,
                                                              random_state=self.rand_seed)
        negative_evidenct_text = self.clean_text(negative_evidenct_sample["evidences"].to_list()[0])
        
        positive_encoding = self.tokenizer(text=claim_text,
                                           text_pair=positive_evidence_text,
                                           add_special_tokens=True,
                                           padding="max_length",
                                           truncation="longest_first",
                                           max_length=self.max_padding_length,
                                           return_tensors="pt")
        
        negative_encoding = self.tokenizer(text=claim_text,
                                           text_pair=negative_evidenct_text,
                                           add_special_tokens=True,
                                           padding="max_length",
                                           truncation="longest_first",
                                           max_length=self.max_padding_length,
                                           return_tensors="pt")
        
        positive_passage = RerankerPassage(input_ids=positive_encoding.input_ids,
                                           segments=positive_encoding.token_type_ids,
                                           attn_mask=positive_encoding.attention_mask)
       
        negative_passage = RerankerPassage(input_ids=negative_encoding.input_ids,
                                           segments=negative_encoding.token_type_ids,
                                           attn_mask=negative_encoding.attention_mask)
        
        return RerankerSample(positive_passage=positive_passage,
                              negative_passage=negative_passage)
    
    
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
    
 
        
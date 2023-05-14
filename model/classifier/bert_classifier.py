import json
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch import Tensor as T
from torch import nn
import transformers
from transformers import BertModel
from .bert_classifier_dataset import BertClassifierDataset

CLASS_TO_IDX = {"SUPPORTS": 0, "REFUTES": 1, "NOT_ENOUGH_INFO": 2, "DISPUTED": 3}
IDX_TO_CLASS = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT_ENOUGH_INFO", 3: "DISPUTED"}

class BertClassifier(nn.Module):
    
    def __init__(self) -> None:
        super(BertClassifier, self).__init__()
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # suppress undesirable warning message
        transformers.logging.set_verbosity_error() 
        
        self.bert_layer = BertModel.from_pretrained("bert-base-uncased").to(self.device)
        hidden_size = self.bert_layer.config.hidden_size
        self.linear_layer = nn.Linear(hidden_size, hidden_size).to(self.device)
        self.activation = nn.Tanh().to(self.device)
        self.linear_output = nn.Linear(hidden_size, 4).to(self.device)
        
        
    def forward(self, input_ids: T, token_type_ids: T, attention_mask: T, return_dict: bool=True):
        
        x = self.bert_layer(input_ids=input_ids,
                              token_type_ids=token_type_ids,
                              attention_mask=attention_mask,
                              return_dict=return_dict).pooler_output
        
        x = self.linear_layer(x)
        x = self.activation(x)
        logit = self.linear_output(x)
        
        return logit


    def predict(self, claim_dataset: BertClassifierDataset, batch_size: int, output_file_path: str=None):
    
        self.eval()
        
        predictions = {}
        claim_dataloader = DataLoader(claim_dataset,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      num_workers=4,
                                      collate_fn=claim_dataset.predict_collate_fn)
    
        for batch_claim_sample in tqdm(claim_dataloader):
            text_sequence = batch_claim_sample.text_sequence
            query_tag = batch_claim_sample.query_tag
                        
            with torch.no_grad():
                
                text_sequence_input_ids = text_sequence.input_ids.squeeze(1).to(self.device)
                text_sequence_segments = text_sequence.segments.squeeze(1).to(self.device)
                text_sequence_attn_mask = text_sequence.attn_mask.squeeze(1).to(self.device)
                
                logit = self.__call__(input_ids=text_sequence_input_ids,
                                      token_type_ids=text_sequence_segments,
                                      attention_mask=text_sequence_attn_mask)    

                predict_idxs = torch.argmax(logit, dim=-1)
                
                for tag, predict_idx in zip(query_tag, predict_idxs):
                    predictions[tag] = IDX_TO_CLASS[predict_idx.tolist()]
            
            del text_sequence_input_ids, text_sequence_segments, text_sequence_attn_mask, logit, predict_idxs       
        
        self.train()
        if output_file_path is not None:
            print("Exporting...")
            f_out = open(output_file_path, 'w')
            json.dump(predictions, f_out)
            f_out.close()
            print('\033[1A', end='\x1b[2K')
            print("File Exported.")
        
        print("Prediction Done!")
        return predictions
        
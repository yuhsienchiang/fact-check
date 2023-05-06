import torch
from torch import Tensor as T
from torch import nn
from transformers import BertModel


class BertClassifier(nn.Module):
    
    def __init__(self) -> None:
        super(BertClassifier, self).__init__()
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.bert_layer = BertModel.from_pretrained("bert-base-uncased").to(device)
        hidden_size = self.bert_layer.config.hidden_size
        self.linear_layer = nn.Linear(hidden_size, 4).to(device)
        
        
    def forward(self, input_ids: T, attention_mask: T, return_dict: bool=True):
        
        _sequence, pooler_out, _hidden = self.bert_layer(input_ids=input_ids,
                                                         attention_mask=attention_mask,
                                                         return_dict=return_dict)
        
        logit = self.linear_layer(pooler_out)
        
        return logit
        
       
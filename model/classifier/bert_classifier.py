import torch
from torch import Tensor as T
from torch import nn
from transformers import BertModel

CLASS_TO_IDX = {"SUPPORTS": 0, "REFUTES": 1, "NOT_ENOUGH_INFO": 2, "DISPUTED": 3}
IDX_TO_CLASS = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT_ENOUGH_INFO", 3: "DISPUTED"}

class BertClassifier(nn.Module):
    
    def __init__(self) -> None:
        super(BertClassifier, self).__init__()
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.bert_layer = BertModel.from_pretrained("bert-base-uncased").to(device)
        hidden_size = self.bert_layer.config.hidden_size
        self.linear_layer = nn.Linear(hidden_size, hidden_size).to(device)
        self.activation = nn.Tanh()
        self.linear_output = nn.Linear(hidden_size, 4)
        
        
    def forward(self, input_ids: T, token_type_ids: T, attention_mask: T, return_dict: bool=True):
        
        _sequence, pooler_out, _hidden = self.bert_layer(input_ids=input_ids,
                                                         token_type_ids=token_type_ids,
                                                         attention_mask=attention_mask,
                                                         return_dict=return_dict)
        
        x = self.linear_layer(pooler_out)
        x = self.activation(x)
        logit = self.linear_output(x)
        
        return logit
    
    def predict(self, input_ids: T, attention_mask: T, return_dict: bool=True):
        
        self.eval()
        logit = self.__call__(input_ids=input_ids, 
                              attention_mask=attention_mask, 
                              return_dict=return_dict)
        
        probability = nn.Softmax(logit, dim=1)
        
        return IDX_TO_CLASS[probability.argmax(1)]
        
       
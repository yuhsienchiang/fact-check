import torch
from torch import Tensor as T
from torch import nn
from transformers import BertModel

from .reranker_dataset import RerankerPassage


class Reranker(nn.Module):
    
    def __init__(self) -> None:
        super(Reranker, self).__init__()
        
        device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
        
        self.encoder_layer = BertModel.from_pretrained("bert-base-uncased").to(device)
        hidden_size = self.encoder_layer.config.hidden_size
        self.linear_layer = nn.Linear(hidden_size, hidden_size).to(device)
        self.activation = nn.Tanh()
        self.linear_output = nn.Linear(hidden_size, 2)
        
        
    def forward(self, input_ids: T, token_type_ids :T, attention_mask: T, return_dict: bool=True):
        
        input_shape = input_ids.shape
        if len(input_shape) == 3: 
            batch_size, vec_num, _vec_len = input_shape
            input_ids = torch.flatten(input_ids, 0, 1)
            token_type_ids = torch.flatten(token_type_ids, 0, 1)
            attention_mask = torch.flatten(attention_mask, 0, 1)
            
        out = self.encoder_layer(input_ids=input_ids,
                                 token_type_ids=token_type_ids,
                                 attention_mask=attention_mask,
                                 return_dict=return_dict)
        
        x = self.linear_layer(out.pooler_output)
        x = self.activation(x)
        logit = self.linear_output(x)
        
        if len(input_shape) == 3:
            logit = torch.unflatten(logit, 0, (batch_size, 1))
        
        return logit

def predict(reranker: Reranker, x :RerankerPassage, return_type: str="idx"):
    
    reranker.eval()
    
    with torch.no_grad():
        logit = reranker(input_ids= x.input_ids,
                         token_type_ids=x.segments,
                         attention_mask=x.attn_mask,
                         return_dict=True)
        
    reranker.train()
    
    if return_type == 'prob':
        return logit
    else:
        return torch.argmax(logit, dim=-1)
    
    
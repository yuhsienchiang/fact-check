from typing import Tuple

import torch
from torch import nn
from torch import Tensor as T
from .bert_encoder import BertEncoder

class BiEncoder(nn.Module):
    def __init__(self, query_model: BertEncoder, evid_model: BertEncoder) -> None:
        super(BiEncoder, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.query_model = query_model.to(device)
        self.evid_model = evid_model.to(device)
        
    def forward(self, query_ids, query_segment, query_attn_mask, evid_ids, evid_segment, evid_attn_mask) -> Tuple[T, T]:
        
        shape = evid_ids.shape
        if len(shape) == 3: 
            batch_size, vec_num, vec_len = shape
            query_ids = torch.flatten(query_ids, 0, 1)
            query_segment = torch.flatten(query_segment, 0, 1)
            query_attn_mask = torch.flatten(query_attn_mask, 0, 1)
            evid_ids = torch.flatten(evid_ids, 0, 1)
            evid_segment = torch.flatten(evid_segment, 0, 1)
            evid_attn_mask = torch.flatten(evid_attn_mask, 0, 1)
        
        
        _query_seq, query_pooler_out, _query_hidden = self.get_representation(sub_model=self.query_model,
                                                                              ids=query_ids,
                                                                              segments=query_segment,
                                                                              attent_mask=query_attn_mask)
        
        _evid_seq, evid_pooler_out, _evid_hidden = self.get_representation(sub_model=self.evid_model,
                                                                           ids=evid_ids,
                                                                           segments=evid_segment,
                                                                           attent_mask=evid_attn_mask)
        
        if len(shape) == 3:
            query_pooled_out = torch.unflatten(query_pooler_out, 0, (batch_size, 1))
            evid_pooled_out = torch.unflatten(evid_pooler_out, 0, (batch_size, vec_len))
        
        return query_pooler_out, evid_pooler_out
    
    def get_representation(self, sub_model: BertEncoder=None, ids: T=None, segments: T=None, attent_mask: T=None) -> Tuple[T, T, T]:
        # make sure the model is add_pooling_layer = True, return_dict=True
        
        if sub_model.training:
            out = sub_model(input_ids=ids,
                            token_type_ids=segments,
                            attention_mask=attent_mask)

        else:
            with torch.no_grad():
                out = sub_model(input_ids=ids,
                                token_type_ids=segments,
                                attention_mask=attent_mask)
        
        sequence = out.last_hidden_state
        pooler_output = out.pooler_output
        hidden_state = out.hidden_states
        
        if sub_model.training:
            sequence.requires_grad = True
            pooler_output.requires_grad = True
        else:
            sequence.requires_grad = False
            pooler_output.requires_grad = False
                
        hidden_state.requires_grad = False
        
        return sequence, pooler_output, hidden_state
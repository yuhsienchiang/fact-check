from typing import Tuple

import torch
from torch import nn
from torch import Tensor as T

class BiEncoder(nn.Module):
    def __init__(self, query_model: nn.Module, evid_model: nn.Module=None) -> None:
        super(BiEncoder, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.encoder_0 = query_model.to(device)
        self.encoder_1 = evid_model.to(device) if evid_model else None
        

    def forward(self, query_ids, query_segment, query_attn_mask, evid_ids, evid_segment, evid_attn_mask) -> Tuple[T, T]:
        
        input_shape = evid_ids.shape
        if len(input_shape) == 3: 
            batch_size, vec_num, _vec_len = input_shape
            query_ids = torch.flatten(query_ids, 0, 1)
            query_segment = torch.flatten(query_segment, 0, 1)
            query_attn_mask = torch.flatten(query_attn_mask, 0, 1)
            evid_ids = torch.flatten(evid_ids, 0, 1)
            evid_segment = torch.flatten(evid_segment, 0, 1)
            evid_attn_mask = torch.flatten(evid_attn_mask, 0, 1)
        
        
        _query_seq, query_pooler_out, _query_hidden = self.get_representation(sub_model=self.encoder_0,
                                                                              ids=query_ids,
                                                                              segments=query_segment,
                                                                              attent_mask=query_attn_mask)
        
        if self.encoder_1:
            _evid_seq, evid_pooler_out, _evid_hidden = self.get_representation(sub_model=self.encoder_1,
                                                                               ids=evid_ids,
                                                                               segments=evid_segment,
                                                                               attent_mask=evid_attn_mask)
        else:
            _evid_seq, evid_pooler_out, _evid_hidden = self.get_representation(sub_model=self.encoder_0,
                                                                               ids=evid_ids,
                                                                               segments=evid_segment,
                                                                               attent_mask=evid_attn_mask)
        
        if len(input_shape) == 3:
            query_pooler_out = torch.unflatten(query_pooler_out, 0, (batch_size, 1))
            evid_pooler_out = torch.unflatten(evid_pooler_out, 0, (batch_size, vec_num))
        
        return query_pooler_out, evid_pooler_out

    
    def get_representation(self, sub_model: nn.Module=None, ids: T=None, segments: T=None, attent_mask: T=None) -> Tuple[T, T, T]:
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
        
        return sequence, pooler_output, hidden_state
    
    
    def encode_query(self, input_ids: T, segments: T, attent_mask: T):
        
        self.encoder_0.eval()
        with torch.no_grad():
            out = self.encoder_0(input_ids=input_ids,
                                 token_type_ids=segments,
                                 attention_mask=attent_mask)
        
        self.encoder_o.train()
        return out.pooler_output

    
    def encode_evidence(self, input_ids: T, segments: T, attent_mask: T):
        
        if self.encoder_1 is not None:
            
            self.encoder_1.eval()
            with torch.no_grad():
                out = self.encoder_1(input_ids=input_ids,
                                     token_type_ids=segments,
                                     attention_mask=attent_mask)
                
                self.encoder_1.train()
                return out.pooler_output
            
        else:
            return self.encode_query(input_ids=input_ids,
                                     segments=segments,
                                     attent_mask=attent_mask)
            
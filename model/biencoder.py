from typing import Tuple

import torch
from torch import nn
from torch import Tensor as T
from bert_encoder import BertEncoder

class BiEncoder(nn.Module):
    def __init__(self, query_model: BertEncoder, evid_model: BertEncoder) -> None:
        super(BiEncoder, self).__init__()
        
        self.query_model = query_model
        self.evid_model = evid_model
        
    def forward(self, query_ids, query_segment, query_attent_mask, evid_ids, evid_segment, evid_attent_mask) -> Tuple[T, T]:
        
        _query_seq, query_pooled_out, _query_hidden = self.get_representation(sub_model=self.query_model,
                                                                              ids=query_ids,
                                                                              segments=query_segment,
                                                                              attent_mask=query_attent_mask)
        
        _evid_seq, evid_pooled_out, _evid_hidden = self.get_representation(sub_model=self.evid_model,
                                                                           ids=evid_ids,
                                                                           segments=evid_segment,
                                                                           attent_mask=evid_attent_mask)
        
        return query_pooled_out, evid_pooled_out
    
    def get_representation(sub_model: nn.Module, ids: T, segments: T, attent_mask: T) -> Tuple[T, T, T]:
        # make sure the model is add_pooling_layer = True, return_dict=True
        
        if sub_model.training:
            out = sub_model(input_ids=ids,
                            token_type_ids=segments,
                            attent_mask=attent_mask)

        else:
            with torch.no_grad():
                out = sub_model(input_ids=ids,
                                token_type_ids=segments,
                                attent_mask=attent_mask)
        
        sequence = out.last_hidden_state
        pooled_output = out.pooled_output
        hidden_state = out.hidden_states
        
        return sequence, pooled_output, hidden_state
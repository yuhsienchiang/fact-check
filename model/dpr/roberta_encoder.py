from typing import Tuple

from torch import Tensor as T
from transformers import RobertaModel, RobertaConfig, RobertaTokenizer


class RoBertaEncoder(RobertaModel):
    
    def __init__(self, config: RobertaConfig=None, add_pooling_layer=True) -> None:
        if config is None:
            super(RoBertaEncoder, self).__init__(RobertaConfig.from_pretrained("roberta-base"), 
                                                 add_pooling_layer)
            super().from_pretrained("roberta-base")
        else:
            super(RoBertaEncoder, self).__init__(config, add_pooling_layer)
    
    def forward(self, input_ids: T, attention_mask: T, token_type_ids: T, return_dict: bool=True) -> Tuple[T]:
        
        return super().forward(input_ids=input_ids,
                               attention_mask=attention_mask, 
                               token_type_ids=token_type_ids,
                               return_dict=return_dict)
        

def get_roberta_encoder_component(config: RobertaConfig=None, add_pooling_layer: bool=True, num_encoder: int=1):
    
    encoder_list = []
    for _ in range(num_encoder):
        encoder = RobertaModel(config=config, add_pooling_layer=add_pooling_layer)
        encoder_list.append(encoder)
        
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    
    return encoder_list, tokenizer
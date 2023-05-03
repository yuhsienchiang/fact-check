from typing import Tuple

from torch import Tensor as T
from transformers import BertModel, BertConfig

class BertEncoder(BertModel):
    
    def __init__(self, config :BertConfig=None, add_pooling_layer :bool=True) -> None:

        if config is None:    
            super(BertEncoder, self).__init__(BertConfig.from_pretrained("bert-base-uncased"),
                                              add_pooling_layer)
            super().from_pretrained("bert-base-uncased")
        else:
            super(BertEncoder, self).__init__(config, add_pooling_layer)

        
    def forward(self, input_ids: T, attention_mask: T, token_type_ids: T, return_dict: bool=True) -> Tuple[T]:
        
        output = super().forward(input_ids=input_ids, 
                                 attention_mask=attention_mask, 
                                 token_type_ids=token_type_ids, 
                                 return_dict=return_dict)
        
        return output
    
    

    
from typing import Tuple

from torch import Tensor as T
import transformers
from transformers import BertModel, BertConfig, BertTokenizer

class BertEncoder(BertModel):
    
    def __init__(self, config :BertConfig=None, add_pooling_layer :bool=True) -> None:
        
        # suppress undesirable warning message
        transformers.logging.set_verbosity_error() 

        if config is None:    
            super(BertEncoder, self).__init__(BertConfig.from_pretrained("bert-base-uncased"),
                                              add_pooling_layer)
            super().from_pretrained("bert-base-uncased")
        else:
            super(BertEncoder, self).__init__(config, add_pooling_layer)

        
    def forward(self, input_ids: T, attention_mask: T, token_type_ids: T, return_dict: bool=True) -> Tuple[T]:
        
        return super().forward(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               return_dict=return_dict)
    
    
def get_bert_encoder_component(config: BertConfig=None, add_pooling_layer:bool=True, num_encoder: int=1):
    
    encoder_list = []
    for _ in range(num_encoder):
        encoder = BertEncoder(config=config, add_pooling_layer=add_pooling_layer)
        encoder_list.append(encoder)
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    return encoder_list, tokenizer
    
    
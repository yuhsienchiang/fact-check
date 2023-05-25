import json
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch import Tensor as T
from torch import nn
import transformers
from transformers import AutoModel
from ...data.classifier_dataset import ClassifierDataset
from ...data.utils import class_label_conv


class Classifier(nn.Module):
    def __init__(self, model_type: str = "bert-base-uncased") -> None:
        super(Classifier, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # suppress undesirable warning message
        transformers.logging.set_verbosity_error()

        self.PLM_layer = AutoModel.from_pretrained(model_type).to(self.device)
        hidden_size = self.bert_layer.config.hidden_size
        self.linear_layer = nn.Linear(hidden_size, hidden_size).to(self.device)
        self.activation = nn.Tanh().to(self.device)
        self.linear_output = nn.Linear(hidden_size, 4).to(self.device)

    def forward(
        self,
        input_ids: T,
        attention_mask: T,
        token_type_ids: T = None,
        return_dict: bool = True,
    ):
        if token_type_ids is not None:
            x = self.PLM(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                return_dict=return_dict,
            ).pooler_output
        else:
            x = self.PLM(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=return_dict,
            ).pooler_output

        x = self.linear_layer(x)
        x = self.activation(x)
        logit = self.linear_output(x)

        return logit

    def predict(
        self,
        claim_dataset: ClassifierDataset,
        batch_size: int,
        output_file_path: str = None,
    ):
        self.eval()

        predictions = {}
        claim_dataloader = DataLoader(
            claim_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=claim_dataset.predict_collate_fn,
        )

        for batch_claim_sample in tqdm(claim_dataloader):
            text_sequence = batch_claim_sample.text_sequence
            query_tag = batch_claim_sample.query_tag

            with torch.no_grad():
                text_sequence_input_ids = text_sequence.input_ids.squeeze(1).to(
                    self.device
                )
                text_sequence_segments = (
                    text_sequence.segments.squeeze(1).to(self.device)
                    if text_sequence.segments is not None
                    else None
                )
                text_sequence_attn_mask = text_sequence.attn_mask.squeeze(1).to(
                    self.device
                )

                logit = self.__call__(
                    input_ids=text_sequence_input_ids,
                    token_type_ids=text_sequence_segments,
                    attention_mask=text_sequence_attn_mask,
                )

                predict_idxs = torch.argmax(logit, dim=-1)

                for tag, predict_idx in zip(query_tag, predict_idxs):
                    predictions[tag] = class_label_conv(predict_idx.tolist())

            del text_sequence_input_ids, text_sequence_attn_mask, logit, predict_idxs
            if text_sequence_segments is not None:
                del text_sequence_segment

        self.train()
        if output_file_path is not None:
            print("Exporting...")
            f_out = open(output_file_path, "w")
            json.dump(predictions, f_out)
            f_out.close()
            print("\033[1A", end="\x1b[2K")
            print("File Exported.")

        print("Prediction Done!")
        return predictions

import json
import pandas as pd
import torch
from torch.utils.data import Dataset
import transformers
from transformers import PreTrainedTokenizer, AutoTokenizer, BertTokenizer
from collections import namedtuple

from data.utils import load_data, class_label_conv

ClassifierSample = namedtuple(
    "ClassifierSample",
    ["query_tag", "query_text", "query_label", "evid_tag", "evid_text"],
)
ClassifierPassage = namedtuple(
    "ClassifierPassage", ["input_ids", "token_type_ids", "attention_mask"]
)
ClassifierTrainSample = namedtuple(
    "ClassifierTrainSample", ["query_tag", "query_label", "text_sequence"]
)
ClassifierPredictSample = namedtuple(
    "ClassifierPredictSample", ["query_tag", "text_sequence"]
)


class ClassifierDataset(Dataset):
    def __init__(
        self,
        claim_file_path: str,
        data_type: str,
        evidence_file_path: str = None,
        tokenizer: PreTrainedTokenizer = None,
        lower_case: bool = False,
        max_padding_length: int = 12,
        rand_seed: int = None,
    ) -> None:
        super(ClassifierDataset, self).__init__()

        self.claim_file_path = claim_file_path
        self.evidence_file_path = evidence_file_path

        self.data_type = data_type

        self.tokenizer = (
            tokenizer
            if tokenizer
            else AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=False)
        )

        self.lower_case = lower_case
        self.max_padding_length = max_padding_length
        self.rand_seed = rand_seed

        self.raw_claim_data = None
        self.raw_evidence_data = None

        self.claim_data = None
        self.evidences_data = None

        # suppress undesirable warning message
        transformers.logging.set_verbosity_error()

        self._load_data()

    def __len__(self) -> int:
        return len(self.raw_claim_data)

    def __getitem__(self, idx):
        data = self.claim_data.iloc[idx]

        query_tag = data["tag"]
        query_text = self.clean_text(data["claim_text"])
        query_label = (
            class_label_conv(data["claim_label"]) if self.data_type == "train" else None
        )

        evidence_tag = data["evidence_tag"]
        evidence_text = [
            self.clean_text(evid, lower_case=self.lower_case)
            for evid in map(self.raw_evidence_data.get, evidence_tag)
        ]

        return ClassifierSample(
            query_tag=query_tag,
            query_label=query_label,
            query_text=query_text,
            evid_tag=evidence_tag,
            evid_text=evidence_text,
        )

    def _load_data(self) -> None:
        raw_claim_data, raw_evidence_data, claim_data, evidence_data = load_data(
            claim_data_path=self.claim_file_path,
            evidence_data_path=self.evidence_file_path,
            data_type=self.data_type,
        )
        self.raw_claim_data = raw_claim_data
        self.raw_evidence_data = raw_evidence_data
        self.claim_data = claim_data
        self.evidences_data = evidence_data

    def clean_text(self, context: str, lower_case: bool = False) -> str:
        return context.lower() if lower_case else context

    def train_collate_fn(self, batch):
        batch_text_sequence_input_ids = []
        batch_text_sequence_token_type_ids = []
        batch_text_sequence_attention_mask = []
        batch_label = []
        batch_tag = []

        for batch_sample in batch:
            claim_text = batch_sample.query_text
            evidence_text = batch_sample.evid_text
            label = batch_sample.query_label
            tag = batch_sample.query_tag

            input_sequence = self.tokenizer.sep_token.join([claim_text] + evidence_text)

            encoding = self.tokenizer(
                text=input_sequence,
                add_special_tokens=True,
                padding="max_length",
                truncation="longest_first",
                max_length=self.max_padding_length,
                return_tensors="pt",
            )

            batch_text_sequence_input_ids.append(encoding.input_ids)
            batch_text_sequence_attention_mask.append(encoding.attention_mask)
            if isinstance(self.tokenizer, BertTokenizer):
                batch_text_sequence_token_type_ids.append(encoding.token_type_ids)

            batch_label.append(label)
            batch_tag.append(tag)

            text_sequences_passage = ClassifierPassage(
                input_ids=torch.stack(batch_text_sequence_input_ids, dim=0),
                token_type_ids=torch.stack(batch_text_sequence_token_type_ids, dim=0)
                if batch_text_sequence_token_type_ids
                else None,
                attention_mask=torch.stack(batch_text_sequence_attention_mask, dim=0),
            )

        return ClassifierTrainSample(
            text_sequence=text_sequences_passage,
            query_label=torch.tensor(batch_label),
            query_tag=batch_tag,
        )

    def predict_collate_fn(self, batch):
        batch_text_sequence_input_ids = []
        batch_text_sequence_token_type_ids = []
        batch_text_sequence_attention_mask = []
        batch_tag = []

        for batch_sample in batch:
            claim_text = batch_sample.query_text
            evidence_text = batch_sample.evid_text
            tag = batch_sample.query_tag

            input_sequence = self.tokenizer.sep_token.join([claim_text] + evidence_text)

            encoding = self.tokenizer(
                text=input_sequence,
                add_special_tokens=True,
                padding="max_length",
                truncation="longest_first",
                max_length=self.max_padding_length,
                return_tensors="pt",
            )

            batch_text_sequence_input_ids.append(encoding.input_ids)
            batch_text_sequence_attention_mask.append(encoding.attention_mask)
            if isinstance(self.tokenizer, BertTokenizer):
                batch_text_sequence_token_type_ids.append(encoding.token_type_ids)

            batch_tag.append(tag)

            text_sequences_passage = ClassifierPassage(
                input_ids=torch.stack(batch_text_sequence_input_ids, dim=0),
                token_type_ids=torch.stack(batch_text_sequence_token_type_ids, dim=0)
                if batch_text_sequence_token_type_ids
                else None,
                attention_mask=torch.stack(batch_text_sequence_attention_mask, dim=0),
            )

        return ClassifierPredictSample(
            text_sequence=text_sequences_passage, query_tag=batch_tag
        )

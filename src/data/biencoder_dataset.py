import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, AutoTokenizer
from collections import namedtuple

from src.utils.utils import load_data, clean_text

BiEncoderSample = namedtuple(
    "BiEncoderSample",
    ["query_tag", "query_label", "query_text", "evid_tag", "evid_text"],
)
BiEncoderPassage = namedtuple(
    "BiEncoderPassage", ["tag", "input_ids", "attention_mask"]
)
BiEncoderTrainSample = namedtuple(
    "BiEncoderTrainSample", ["query", "evid", "is_positive"]
)
BiEncoderEvaluateSample = namedtuple("BiEncoderEvaluateSample", ["query", "evid_tag"])
BiEncoderPredictSample = namedtuple("BiEncoderPredictSample", ["query", "query_tag"])


class BiEncoderDataset(Dataset):
    def __init__(
        self,
        claim_file_path: str,
        data_type: str,
        evidence_file_path: str = None,
        tokenizer: PreTrainedTokenizer = None,
        lower_case: bool = False,
        max_padding_length: int = 12,
        evidence_num: int = None,
        rand_seed: int = None,
    ) -> None:
        super(BiEncoderDataset, self).__init__()

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

        self._load_data()

        if self.data_type == "train":
            self.max_positive_num = (
                self.claim_data["evidence"].apply(lambda x: len(x)).max()
            )
            self.evidence_num = (
                max(evidence_num, self.max_positive_num)
                if evidence_num is not None
                else self.max_positive_num
            )

    def __len__(self) -> int:
        return len(self.raw_claim_data)

    def __getitem__(self, idx):
        # fetch sample
        data = self.claim_data.iloc[idx]

        query_tag = data["tag"]
        # extract query text and apply cleaning
        query_text = clean_text(data["claim_text"], lower_case=self.lower_case)

        query_label = data["label"] if self.data_type == "train" else None

        # extract evidence_tag
        evidence_tag = data["evidence"] if self.data_type == "train" else None
        # extract evidence text and apply cleaning
        evidence_text = (
            [
                clean_text(evid, lower_case=self.lower_case)
                for evid in map(self.raw_evidence_data.get, evidence_tag)
            ]
            if self.data_type == "train"
            else None
        )

        return BiEncoderSample(
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

    def train_collate_fn(self, batch):
        batch_query_tag = []
        batch_query_input_ids = []
        batch_query_attention_mask = []
        batch_evid_tag = []
        batch_evid_input_ids = []
        batch_evid_attention_mask = []
        batch_is_positive = []

        for batch_sample in batch:
            query_text = batch_sample.query_text
            evid_text = batch_sample.evid_text

            query_tokenized = self.tokenizer(
                text=query_text,
                add_special_tokens=True,
                padding="max_length",
                truncation="longest_first",
                max_length=self.max_padding_length,
                return_tensors="pt",
            )

            batch_query_tag.append(batch.query_tag)
            batch_query_input_ids.append(query_tokenized.input_ids)
            batch_query_attention_mask.append(query_tokenized.attention_mask)

            # prepare positive evid text
            is_positive = len(evid_text)
            batch_is_positive.append(is_positive)

            # prepare negative evid text
            # sample (evidence num - positive evid num) negative evidence
            # ensure the total evidence amout for each sample is the same
            negative_evid_sample = self.evidences_data.sample(
                n=self.evidence_num - is_positive, random_state=self.rand_seed
            )["evidence"].tolist()
            negative_evid_textset = [
                clean_text(neg_evidence, lower_case=self.lower_case)
                for neg_evidence in negative_evid_sample
            ]

            # combine positive and negative evid into a single textset
            evid_textset = evid_text + negative_evid_textset

            # tokenized evidence text
            evid_tokenized = self.tokenizer(
                text=evid_textset,
                add_special_tokens=True,
                padding="max_length",
                truncation="longest_first",
                max_length=self.max_padding_length,
                return_tensors="pt",
            )

            batch_evid_tag.append(batch.eivd_tag)
            batch_evid_input_ids.append(evid_tokenized.input_ids)
            batch_evid_attention_mask.append(evid_tokenized.attention_mask)

        batch_query = BiEncoderPassage(
            input_ids=torch.stack(batch_query_input_ids, dim=0),
            attention_mask=torch.stack(batch_query_attention_mask, dim=0),
        )

        batch_evid = BiEncoderPassage(
            input_ids=torch.stack(batch_evid_input_ids, dim=0),
            attention_mask=torch.stack(batch_evid_attention_mask, dim=0),
        )

        return BiEncoderTrainSample(
            query=batch_query,
            evid=batch_evid,
            is_positive=torch.tensor(batch_is_positive),
        )

    def evaluate_collate_fn(self, batch):
        batch_query_input_ids = []
        batch_query_attention_mask = []
        batch_evid_tag = []

        for batch_sample in batch:
            query_text = batch_sample.query_text
            evid_tag = batch_sample.evid_tag

            query_tokenized = self.tokenizer(
                text=query_text,
                add_special_tokens=True,
                padding="max_length",
                truncation="longest_first",
                max_length=self.max_padding_length,
                return_tensors="pt",
            )

            batch_query_input_ids.append(query_tokenized.input_ids)
            batch_query_attention_mask.append(query_tokenized.attention_mask)

            batch_evid_tag.append(evid_tag)

        batch_query = BiEncoderPassage(
            input_ids=torch.stack(batch_query_input_ids, dim=0),
            attention_mask=torch.stack(batch_query_attention_mask, dim=0),
        )

        return BiEncoderEvaluateSample(query=batch_query, evid_tag=batch_evid_tag)

    def predict_collate_fn(self, batch):
        batch_query_input_ids = []
        batch_query_attention_mask = []
        batch_query_tag = []

        for batch_sample in batch:
            query_text = batch_sample.query_text
            query_tag = batch_sample.query_tag

            query_tokenized = self.tokenizer(
                text=query_text,
                add_special_tokens=True,
                padding="max_length",
                truncation="longest_first",
                max_length=self.max_padding_length,
                return_tensors="pt",
            )

            batch_query_input_ids.append(query_tokenized.input_ids)
            batch_query_attention_mask.append(query_tokenized.attention_mask)

            batch_query_tag.append(query_tag)

        batch_query = BiEncoderPassage(
            input_ids=torch.stack(batch_query_input_ids, dim=0),
            attention_mask=torch.stack(batch_query_attention_mask, dim=0),
        )

        return BiEncoderPredictSample(query=batch_query, query_tag=batch_query_tag)

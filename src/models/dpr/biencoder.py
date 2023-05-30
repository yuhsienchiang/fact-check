from typing import Tuple
import json
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor as T
from torch.utils.data import DataLoader
from data.evidence_dataset import EvidenceDataset
from data.biencoder_dataset import BiEncoderDataset


class BiEncoder(nn.Module):
    def __init__(
        self,
        query_model: nn.Module,
        evid_model: nn.Module = None,
        similarity_func_type: str = None,
    ) -> None:
        super(BiEncoder, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.encoder_0 = query_model.to(self.device)
        self.encoder_1 = evid_model.to(self.device) if evid_model is not None else None

        self.similarity_func_type = similarity_func_type

    def forward(
        self, query_ids, query_attn_mask, evid_ids, evid_attn_mask
    ) -> Tuple[T, T]:
        _query_seq, query_pooler_out, _query_hidden = self.get_representation(
            sub_model=self.encoder_0, ids=query_ids, attent_mask=query_attn_mask
        )

        if self.encoder_1 is not None:
            _evid_seq, evid_pooler_out, _evid_hidden = self.get_representation(
                sub_model=self.encoder_1, ids=evid_ids, attent_mask=evid_attn_mask
            )
        else:
            _evid_seq, evid_pooler_out, _evid_hidden = self.get_representation(
                sub_model=self.encoder_0, ids=evid_ids, attent_mask=evid_attn_mask
            )

        return query_pooler_out, evid_pooler_out

    def get_representation(
        self, sub_model: nn.Module = None, ids: T = None, attent_mask: T = None
    ) -> Tuple[T, T]:
        # make sure the model is add_pooling_layer = True, return_dict=True

        if sub_model.training:
            out = sub_model(input_ids=ids, attention_mask=attent_mask)

        else:
            with torch.no_grad():
                out = sub_model(input_ids=ids, attention_mask=attent_mask)

        sequence = out.last_hidden_state
        pooler_output = out.pooler_output
        hidden_state = out.hidden_states

        return sequence, pooler_output, hidden_state

    def encode_query(self, input_ids: T, attent_mask: T):
        self.encoder_0.eval()
        _sequence, query_pooler_output, _hidden_state = self.get_representation(
            self.encoder_0, ids=input_ids, attent_mask=attent_mask
        )

        self.encoder_0.train()

        return query_pooler_output

    def encode_evidence(self, input_ids: T, attent_mask: T):
        if self.encoder_1 is not None:
            self.encoder_1.eval()

            _seq, evid_pooler_output, _hidden = self.get_representation(
                self.encoder_1, ids=input_ids, attent_mask=attent_mask
            )
            self.encoder_1.train()

        else:
            evid_pooler_output = self.encode_query(
                input_ids=input_ids, attent_mask=attent_mask
            )

        return evid_pooler_output

    def get_evidence_embed(
        self,
        evidence_dataset: EvidenceDataset = None,
        batch_size: int = 64,
        output_file_path: str = None,
    ):
        evidence_embed = []
        evidence_tag = []

        evidence_dataloader = DataLoader(
            evidence_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=evidence_dataset.evidence_collate_fn,
        )

        for batch_sample in tqdm(evidence_dataloader):
            evidence = batch_sample.evidence_tok

            evid_input_ids = evidence.input_ids.squeeze(1).to(self.device)
            evid_attent_mask = evidence.attn_mask.squeeze(1).to(self.device)

            evid_embed = self.encode_evidence(
                input_ids=evid_input_ids, attent_mask=evid_attent_mask
            )

            evid_embed_cpu = evid_embed.cpu()

            del evid_embed, evid_input_ids, evid_attent_mask

            evidence_embed.append(evid_embed_cpu)
            evidence_tag.extend(batch_sample.tag)

        evidence_embed = torch.cat(evidence_embed, dim=0)

        if output_file_path is not None:
            print("Exporting...")
            evid_embed_dict = {
                tag: embed.tolist() for tag, embed in zip(evidence_tag, evidence_embed)
            }
            f_out = open("data/output/embed-evidence.json", "w")
            json.dump(evid_embed_dict, f_out)
            f_out.close()
            print("\033[1A", end="\x1b[2K")
            print("File Exported.")

        evid_embed_dict = {"tag": evidence_tag, "evidence": evidence_embed.tolist()}
        del evidence_tag, evidence_embed
        print("Embedding Done!")
        return evid_embed_dict

    def retrieve(
        self,
        claim_dataset: BiEncoderDataset,
        embed_evid_data: dict,
        batch_size: int = 64,
        k: int = 5,
        predict_output_path: str = None,
    ):
        claim_dataloader = DataLoader(
            claim_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=claim_dataset.predict_collate_fn,
        )

        if self.similarity_func_type == "dot":
            evid_embed = torch.tensor(embed_evid_data["evidence"]).t().to(self.device)
        elif self.similarity_func_type == "cosine":
            evid_embed = (
                F.normalize(torch.tensor(embed_evid_data["evidence"]), p=2, dim=-1)
                .t()
                .to(self.device)
            )

        tags = embed_evid_data["tag"]

        predictions = {}

        for batch_claim_sample in tqdm(claim_dataloader):
            query = batch_claim_sample.query
            query_tags = batch_claim_sample.query_tag

            query_input_ids = query.input_ids.squeeze(1).to(self.device)
            query_attn_mask = query.attn_mask.squeeze(1).to(self.device)

            query_embed = self.encode_query(
                input_ids=query_input_ids, attent_mask=query_attn_mask
            )

            if self.similarity_func_type == "dot":
                similarity_score = torch.mm(query_embed, evid_embed)
            elif self.similarity_func_type == "cosine":
                similarity_score = torch.mm(
                    F.normalize(query_embed, p=2, dim=-1), evid_embed
                )

            top_ks_idxs = torch.topk(similarity_score, k=k, dim=1).indices.tolist()

            for query_tag, top_k_idx in zip(query_tags, top_ks_idxs):
                predictions[query_tag] = [tags[idx] for idx in top_k_idx]

            del (
                query_input_ids,
                query_attn_mask,
                query_embed,
                similarity_score,
                top_ks_idxs,
            )

        del evid_embed

        if predict_output_path is not None:
            print("Exporting...")
            f_out = open(predict_output_path, "w")
            json.dump(predictions, f_out)
            f_out.close()
            print("\033[1A", end="\x1b[2K")
            print("File Exported.")

        print("Retrieve Done!")
        return predictions

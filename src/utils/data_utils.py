import json
import pandas as pd


def load_data(
    claim_data_path: str, evidence_data_path: str = None, data_type: str = "train"
):
    raw_claim_data = json.load(open(claim_data_path))
    raw_evidence_data = (
        json.load(open(evidence_data_path)) if evidence_data_path else None
    )

    if data_type == "train":
        normalized_claim_data = [
            {
                "tag": key,
                "claim_text": value["claim_text"],
                "claim_label": value["claim_label"],
                "evidence": value["evidences"],
            }
            for (key, value) in raw_claim_data.items()
        ]
    else:
        normalized_claim_data = [
            {
                "tag": key,
                "claim_text": value["claim_text"],
            }
            for (key, value) in raw_claim_data.items()
        ]

    normalized_evidence_data = (
        [{"tag": key, "evidence": value} for (key, value) in raw_evidence_data.items()]
        if raw_evidence_data
        else None
    )

    claim_data = pd.json_normalize(normalized_claim_data)
    evidences_data = (
        pd.json_normalize(normalized_evidence_data)
        if normalized_evidence_data
        else None
    )

    return raw_claim_data, raw_evidence_data, claim_data, evidences_data


def clean_text(context: str, lower_case: bool = False) -> str:
    context = context.replace("`", "'")
    context = context.replace(" 's", "'s")
    return context.lower() if lower_case else context
from transformers import AutoModel, AutoTokenizer
from utils.data_utils import load_config
from data.biencoder_dataset import BiEncoderDataset
from data.evidence_dataset import EvidenceDataset
from models.dpr.biencoder import BiEncoder
from models.dpr.biencoder_trainer import BiEncoderTrainer


def run():
    args = load_config("config/biencoder_config.yaml")
    root_path = "../"

    # create tokenizer
    print("Create tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.dataset.tokenizer.type, use_fast=False
    )
    print("Tokenizer created.")

    # Prepare dataset
    # Training dataset
    print("Load datasets...")
    biencoder_train_dataset = BiEncoderDataset(
        claim_file_path=root_path + args.dataset.train_dataset_path,
        data_type="train",
        evidence_file_path=root_path + args.dataset.evidence_dataset_path,
        tokenizer=tokenizer,
        lower_case=args.dataset.tokenizer.lower_case,
        max_padding_length=args.dataset.tokenizer.token_length,
        evidence_num=args.dataset.evidence_num,
        rand_seed=args.dataset.tokenizer.rand_seed,
    )
    # Testing dataset
    biencoder_valid_train_dataset = BiEncoderDataset(
        claim_file_path=root_path + args.dataset.train_dataset_path,
        data_type="predict",
        evidence_file_path=root_path + args.dataset.evidence_dataset_path,
        tokenizer=tokenizer,
        lower_case=args.dataset.tokenizer.lower_case,
        max_padding_length=args.dataset.tokenizer.token_length,
        evidence_num=0,
        rand_seed=args.dataset.tokenizer.rand_seed,
    )

    biencoder_valid_dev_dataset = BiEncoderDataset(
        claim_file_path=root_path + args.dataset.dev_dataset_path,
        data_type="predict",
        evidence_file_path=root_path + args.dataset.evidence_dataset_path,
        tokenizer=tokenizer,
        lower_case=args.dataset.tokenizer.lower_case,
        max_padding_length=args.dataset.tokenizer.token_length,
        evidence_num=0,
        rand_seed=args.dataset.tokenizer.rand_seed,
    )

    biencoder_test_dataset = BiEncoderDataset(
        claim_file_path=root_path + args.dataset.test_dataset_path,
        data_type="predict",
        evidence_file_path=root_path + args.dataset.evidence_dataset_path,
        tokenizer=tokenizer,
        lower_case=args.dataset.tokenizer.lower_case,
        max_padding_length=args.dataset.tokenizer.token_length,
        evidence_num=0,
        rand_seed=args.dataset.tokenizer.rand_seed,
    )

    # Evidence dataset
    evidence_dataset = EvidenceDataset(
        evidence_file_path=root_path + args.dataset.evidence_dataset_path,
        tokenizer=tokenizer,
        lower_case=args.dataset.tokenizer.lower_case,
        max_padding_length=args.dataset.tokenizer.token_length,
        rand_seed=args.dataset.tokenizer.rand_seed,
    )
    print("Datasets loaded.")

    # create model
    print("Create BiEncoder...")
    query_encoder = AutoModel.from_pretrained(args.model.query_encoder_type)
    evidence_encoder = (
        AutoModel.from_pretrained(args.model.evidence_encoder_type)
        if args.model.evidence_encoder_type
        else None
    )
    biencoder = BiEncoder(
        query_model=query_encoder,
        evid_model=evidence_encoder,
        similarity_func_type=args.model.similarity_func,
    )
    print("BiEncoder created.")

    # train model
    print("Create model trainer.")
    biencoder_trainer = BiEncoderTrainer(
        model=biencoder, batch_size=args.train.batch_size
    )

    print("Training starts...")
    biencoder_history = biencoder_trainer.train(
        train_dataset=biencoder_train_dataset,
        shuffle=args.train.shuffle,
        max_epoch=args.train.epoch,
        loss_func_type=args.train.loss_func,
        optimizer_type=args.train.optimiser,
        learning_rate=args.train.learning_rate,
    )

    # Encode evidence
    print("Start embedding evidences....")
    print("Embedding...")
    embed_evid_data = biencoder.get_evidence_embed(
        evidence_dataset=evidence_dataset,
        output_file_path=root_path + args.gen_evid_embeds.output_path,
        batch_size=args.gen_evid_embeds.batch_size,
    )

    # retrieve info
    print("Start retrieve info...")
    retrieve_valid_train_output = biencoder.retrieve(
        biencoder_valid_train_dataset,
        embed_evid_data=embed_evid_data,
        k=args.retrieve.train.retrieve_num,
        batch_size=args.retrieve.train.batch_size,
        predict_output_path=root_path + args.retrieve.train.output_path,
    )

    retrieve_valid_devoutput = biencoder.retrieve(
        biencoder_valid_dev_dataset,
        embed_evid_data=embed_evid_data,
        k=args.retrieve.dev.retrieve_num,
        batch_size=args.retrieve.dev.batch_size,
        predict_output_path=root_path + args.retrieve.dev.output_path,
    )

    retrieve_test_output = biencoder.retrieve(
        biencoder_test_dataset,
        embed_evid_data=embed_evid_data,
        k=args.retrieve.test.retrieve_num,
        batch_size=args.retrieve.test.batch_size,
        predict_output_path=root_path + args.retrieve.test.output_path,
    )


if __name__ == "__main__":
    run()

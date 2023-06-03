from transformers import AutoTokenizer
from utils.data_utils import load_config
from models.classifier.classifier import Classifier
from data.classifier_dataset import ClassifierDataset
from models.classifier.classifier_trainer import ClassifierTrainer


def run():
    args = load_config("./config/classifier_config.yaml")
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
    classifier_train_dataset = ClassifierDataset(
        claim_file_path=root_path + args.dataset.train_dataset_path,
        data_type="train",
        evidence_file_path=root_path + args.dataset.evidence_dataset_path,
        tokenizer=tokenizer,
        lower_case=args.dataset.tokenizer.lower_case,
        max_padding_length=args.dataset.tokenizer.token_length,
        rand_seed=args.dataset.tokenizer.rand_seed,
    )
    # Testing dataset
    classifier_valid_train_dataset = ClassifierDataset(
        claim_file_path=root_path + args.dataset.train_dataset_path,
        data_type="predict",
        evidence_file_path=root_path + args.dataset.evidence_dataset_path,
        tokenizer=tokenizer,
        lower_case=args.dataset.tokenizer.lower_case,
        max_padding_length=args.dataset.tokenizer.token_length,
        rand_seed=args.dataset.tokenizer.rand_seed,
    )

    classifier_valid_dev_dataset = ClassifierDataset(
        claim_file_path=root_path + args.dataset.dev_dataset_path,
        data_type="predict",
        evidence_file_path=root_path + args.dataset.evidence_dataset_path,
        tokenizer=tokenizer,
        lower_case=args.dataset.tokenizer.lower_case,
        max_padding_length=args.dataset.tokenizer.token_length,
        rand_seed=args.dataset.tokenizer.rand_seed,
    )

    classifier_test_dataset = ClassifierDataset(
        claim_file_path=root_path + args.dataset.test_dataset_path,
        data_type="predict",
        evidence_file_path=args.dataset.evidence_dataset_path,
        tokenizer=tokenizer,
        lower_case=args.dataset.tokenizer.lower_case,
        max_padding_length=args.dataset.tokenizer.token_length,
        rand_seed=args.dataset.tokenizer.rand_seed,
    )

    # create model
    print("Create Bert classifier")
    classifier = Classifier(model_type=args.model.type)
    print("Bert classifier created.")

    # train model
    print("Create model trainer.")
    classifier_trainer = ClassifierTrainer(classifier, batch_size=args.train.batch_size)

    print("Training starts...")
    classifier_trainer.train(
        train_dataset=classifier_train_dataset,
        shuffle=args.train.shuffle,
        max_epoch=args.train.epoch,
        loss_func_type=args.train.loss_func,
        optimizer_type=args.train.optimiser,
        learning_rate=args.train.learning_rate,
    )

    # Predict
    print("Start prediction...")
    classification_valid_train_output = classifier.predict(
        classifier_valid_train_dataset,
        batch_size=args.predict.train.batch_size,
        output_file_path=root_path + args.predict.train.output_path
    )

    classification_valid_dev_output = classifier.predict(
        classifier_valid_dev_dataset,
        batch_size=args.predict.dev.batch_size,
        output_file_path=root_path + args.predict.dev.output_path
    )

    classification_test_output = classifier.predict(
        classifier_test_dataset,
        batch_size=args.predict.test.batch_size,
        output_file_path=root_path + args.predict.test.output_path
    )


if __name__ == "__main__":
    run()

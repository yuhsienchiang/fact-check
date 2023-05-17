import time
from transformers import AutoTokenizer

from model.classifier.bert_classifier import BertClassifier
from model.classifier.bert_classifier_dataset import BertClassifierDataset
from model.classifier.bert_classifier_trainer import BertClassifierTrainer



def run():
    
    # create tokenizer
    print("Create tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=False)
    print('\033[1A', end='\x1b[2K')
    print("Tokenizer created.")

    # Prepare dataset
    # Training dataset
    print("Load datasets...")
    classifier_train_dataset = BertClassifierDataset(claim_file_path="./data/train-claims.json",
                                                     data_type="train",
                                                     evidence_file_path="./data/evidence.json",
                                                     tokenizer=tokenizer,
                                                     lower_case=True,
                                                     max_padding_length=512,
                                                     rand_seed=666)
    # Testing dataset
    classifier_valid_train_dataset = BertClassifierDataset(claim_file_path="./data/train-claims.json",
                                                           data_type="predict",
                                                           evidence_file_path="./data/evidence.json",
                                                           predict_evidence_file_path="./data/output/retrieve-train-claims.json",
                                                           tokenizer=tokenizer,
                                                           lower_case=True,
                                                           max_padding_length=512,
                                                           rand_seed=666)
    
    classifier_valid_dev_dataset = BertClassifierDataset(claim_file_path="./data/dev-claims.json",
                                                           data_type="predict",
                                                           evidence_file_path="./data/evidence.json",
                                                           predict_evidence_file_path="./data/output/retrieve-dev-claims.json",
                                                           tokenizer=tokenizer,
                                                           lower_case=True,
                                                           max_padding_length=512,
                                                           rand_seed=666)
    
    classifier_test_dataset = BertClassifierDataset(claim_file_path="./data/test-claims-unlabelled.json",
                                                    data_type="predict",
                                                    evidence_file_path="./data/evidence.json",
                                                    predict_evidence_file_path="./data/output/retrieve-test-claims.json",
                                                    tokenizer=tokenizer,
                                                    lower_case=True,
                                                    max_padding_length=512,
                                                    rand_seed=666)
    
    # create model
    print("Create Bert classifier")
    bert_classifier = BertClassifier(model_type="roberta-base")
    print("Bert classifier created.")
    
    # train model
    print("Create model trainer.")
    bert_classifier_trainer = BertClassifierTrainer(bert_classifier,
                                                    batch_size=4)

    print("Training starts...")
    bert_classifier_trainer.train(train_dataset=classifier_train_dataset,
                                  shuffle=True,
                                  max_epoch=20,
                                  loss_func_type="cross_entropy",
                                  optimizer_type="adam",
                                  learning_rate=1e-5)
    
    # Predict
    print("Start prediction...")
    classification_valid_train_output= bert_classifier.predict(classifier_valid_train_dataset,
                                                               batch_size=4,
                                                               output_file_path="./data/output/classify-dev-claims.json")
    
    classification_valid_dev_output= bert_classifier.predict(classifier_valid_dev_dataset,
                                                             batch_size=4,
                                                             output_file_path="./data/output/classify-dev-claims.json")
    
    classification_test_output= bert_classifier.predict(classifier_test_dataset,
                                                        batch_size=4,
                                                        output_file_path="./data/output/classify-test-claims.json")
    
if __name__ == "__main__":
    run()
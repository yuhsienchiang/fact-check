import argparse
import time
from transformers import BertTokenizer

from model.classifier.bert_classifier import BertClassifier
from model.classifier.bert_classifier_dataset import BertClassifierDataset
from model.classifier.bert_classifier_trainer import BertClassifierTrainer



def run():
    
    # create tokenizer
    print("Create tokenizer...")
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    print('\033[1A', end='\x1b[2K')
    print("Tokenizer created.")

    # Prepare dataset
    # Training dataset
    print("Load datasets...")
    classifier_train_dataset = BertClassifierDataset(claim_file_path="./data/train-claims.json",
                                                     data_type="train",
                                                     evidence_file_path="./data/evidence.json",
                                                     tokenizer=bert_tokenizer,
                                                     lower_case=False,
                                                     max_padding_length=512,
                                                     rand_seed=666)
    # Testing dataset
    classifier_valid_train_dataset = BertClassifierDataset(claim_file_path="./data/train-claims.json",
                                                           data_type="predict",
                                                           evidence_file_path="./data/evidence.json",
                                                           predict_evidence_file_path="./data/output/retrieve-train-claims.json",
                                                           tokenizer=bert_tokenizer,
                                                           lower_case=False,
                                                           max_padding_length=512,
                                                           rand_seed=666)
    
    classifier_valid_dev_dataset = BertClassifierDataset(claim_file_path="./data/dev-claims.json",
                                                           data_type="predict",
                                                           evidence_file_path="./data/evidence.json",
                                                           predict_evidence_file_path="./data/output/retrieve-dev-claims.json",
                                                           tokenizer=bert_tokenizer,
                                                           lower_case=False,
                                                           max_padding_length=512,
                                                           rand_seed=666)
    
    # create model
    print("Create Bert classifier")
    bert_classifier = BertClassifier()
    print('\033[1A', end='\x1b[2K')
    print("Bert classifier created.")
    
    # train model
    print("Create model trainer.")
    bert_classifier_trainer = BertClassifierTrainer(bert_classifier,
                                                    batch_size=4)
    print('\033[1A', end='\x1b[2K')
    print("Training starts...")
    bert_classifier_trainer.train(train_dataset=classifier_train_dataset,
                                  shuffle=True,
                                  max_epoch=20,
                                  loss_func_type="cross_entropy",
                                  optimizer_type="adam",
                                  learning_rate=1e-5)
    
    # Predict
    print("Start prediction...")
    time.sleep(3)
    print('\033[1A', end='\x1b[2K')
    classification_output= bert_classifier.predict(classifier_valid_train_dataset,
                                                   batch_size=4,
                                                   output_file_path="./data/output/classify-train-claims.json")
    
    classification_output= bert_classifier.predict(classifier_valid_dev_dataset,
                                                   batch_size=4,
                                                   output_file_path="./data/output/classify-dev-claims.json")
if __name__ == "__main__":
    run()
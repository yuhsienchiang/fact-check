import argparse
from transformers import BertTokenizer

from model.classifier.bert_classifier import BertClassifier
from model.classifier.bert_classifier_dataset import BertClassifierDataset
from model.classifier.bert_classifier_trainer import BertClassifierTrainer



def run(args):
    
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    classifier_train_dataset = BertClassifierDataset(claim_file_path="./data/train-claims.json",
                                                     data_type="train",
                                                     evidence_file_path="./data/evidence.json",
                                                     tokenizer=bert_tokenizer,
                                                     lower_case=False,
                                                     max_padding_length=512,
                                                     rand_seed=666)
    
    classifier_test_dataset = BertClassifierDataset(claim_file_path="./data/test-claims-unlabelled.json",
                                                    idata_type="predict",
                                                    evidence_file_path="./data/evidence.json",
                                                    predict_evidence_file_path="./data/output/retrieval-claim-prediction.json",
                                                    tokenizer=bert_tokenizer,
                                                    lower_case=False,
                                                    max_padding_length=512,
                                                    rand_seed=666)
    
    bert_classifier = BertClassifier()
    
    bert_classifier_trainer = BertClassifierTrainer(bert_classifier,
                                                    batch_size=4)
    
    bert_classifier_trainer.train(train_dataset=classifier_train_dataset,
                                  shuffle=True,
                                  max_epoch=5,
                                  loss_func_type="cross_entropy",
                                  optimizer_type="adam",
                                  learning_rate=0.00001)
    
    classification_output= bert_classifier.predict(classifier_train_dataset,
                                                   batch_size=4,
                                                   output_file_path=args.predict_output_path)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("--predict_output_path", default=None, type=str, help="prediction output file path")
    args = parser.parse_args()
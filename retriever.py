import argparse
import time
from transformers import BertTokenizer
from model.dpr.biencoder_dataset import BiEncoderDataset
from model.dpr.evidence_dataset import EvidenceDataset
from model.dpr.biencoder import BiEncoder
from model.dpr.encoder.bert_encoder import BertEncoder
from model.dpr.biencoder_trainer import BiEncoderTrainer

def run(args):
    
    # create tokenizer
    print("Create tokenizer...")
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    print('\033[1A', end='\x1b[2K')
    print("Tokenizer created.")

    
    # Prepare dataset
    # Training dataset
    print("Load datasets...")
    biencoder_train_dataset = BiEncoderDataset(claim_file_path="./data/train-claims.json",
                                               data_type="train",
                                               evidence_file_path="./data/evidence.json",
                                               tokenizer=bert_tokenizer,
                                               lower_case=False,
                                               max_padding_length=128,
                                               evidence_num=6,
                                               rand_seed=5)
    # Testing dataset
    biencoder_test_dataset = BiEncoderDataset(claim_file_path="./data/test-claims-unlabelled.json",
                                              data_type="predict",
                                              evidence_file_path="./data/evidence.json",
                                              tokenizer=bert_tokenizer,
                                              lower_case=False,
                                              max_padding_length=128,
                                              evidence_num=0,
                                              rand_seed=5)
    #Evidence dataset
    evidence_dataset = EvidenceDataset(evidence_file_path="./data/evidence.json",
                                       tokenizer=bert_tokenizer,
                                       lower_case=False,
                                       max_padding_length=128,
                                       rand_seed=5)
    print('\033[1A', end='\x1b[2K')
    print("Datasets loaded.")
    
    # create model
    print("Create BiEncoder...")
    query_encoder = BertEncoder(add_pooling_layer=True)
    biencoder = BiEncoder(query_model=query_encoder)
    print('\033[1A', end='\x1b[2K')
    print("BiEncoder created.")
    
    # train model
    print("Create model trainer.")
    biencoder_trainer = BiEncoderTrainer(model=biencoder,
                                         batch_size=8)
    print('\033[1A', end='\x1b[2K')
    print("Training starts...")
    biencoder_history = biencoder_trainer.train(train_dataset=biencoder_train_dataset,
                                                shuffle=True,
                                                max_epoch=5,
                                                loss_func_type="nll_loss",
                                                similarity_func_type="dot",
                                                optimizer_type="adam",
                                                learning_rate=0.0001)
    
    # Encode evidence
    print("Start embedding evidences....")
    time.sleep(3)
    evid_embed_data = biencoder.get_evidence_embed(evidence_dataset=evidence_dataset,
                                                   batch_size=1000,
                                                   output_file_path=args.embed_output_path)

    # retrieve info
    print("Start retrieve info...")
    time.sleep(3)
    print('\033[1A', end='\x1b[2K')
    retrieve_output = biencoder.retrieve(biencoder_test_dataset,
                                         embed_evid_data=evid_embed_data,
                                         k=6,
                                         batch_size =128,
                                         predict_output_path=args.retrieve_output_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("--embed_output_path", default=None, type=str, help="evidence embedding output file path")
    parser.add_argument("--retrieve_output_path", default=None, type=str, help="retrieval data output file path")
    args = parser.parse_args()
    run(args)
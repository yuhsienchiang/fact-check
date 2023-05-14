import time
from transformers import BertModel
from transformers import BertTokenizer
from model.dpr.biencoder_dataset import BiEncoderDataset
from model.dpr.evidence_dataset import EvidenceDataset
from model.dpr.biencoder import BiEncoder
from model.dpr.encoder.bert_encoder import BertEncoder
from model.dpr.biencoder_trainer import BiEncoderTrainer

def run():
    
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
                                               evidence_num=16,
                                               rand_seed=666)
    # Testing dataset
    biencoder_valid_train_dataset = BiEncoderDataset(claim_file_path="./data/train-claims.json",
                                                     data_type="predict",
                                                     evidence_file_path="./data/evidence.json",
                                                     tokenizer=bert_tokenizer,
                                                     lower_case=False,
                                                     max_padding_length=128,
                                                     evidence_num=0,
                                                     rand_seed=666)

    biencoder_valid_dev_dataset = BiEncoderDataset(claim_file_path="./data/dev-claims.json",
                                                   data_type="predict",
                                                   evidence_file_path="./data/evidence.json",
                                                   tokenizer=bert_tokenizer,
                                                   lower_case=False,
                                                   max_padding_length=128,
                                                   evidence_num=0,
                                                   rand_seed=666)
    
    
    #Evidence dataset
    evidence_dataset = EvidenceDataset(evidence_file_path="./data/evidence.json",
                                       tokenizer=bert_tokenizer,
                                       lower_case=False,
                                       max_padding_length=128,
                                       rand_seed=666)
    
    print('\033[1A', end='\x1b[2K')
    print("Datasets loaded.")
    
    # create model
    print("Create BiEncoder...")
    query_encoder = BertModel.from_pretrained("bert-base-uncased")
    biencoder = BiEncoder(query_model=query_encoder, similarity_func_type="dot")
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
                                                max_epoch=20,
                                                loss_func_type="nll_loss",
                                                optimizer_type="adam",
                                                learning_rate=2e-5)
    
    # Encode evidence
    print("Start embedding evidences....")
    time.sleep(3)
    print("Embedding...")
    embed_evid_data = biencoder.get_evidence_embed(evidence_dataset=evidence_dataset,
                                                   batch_size=1000,
                                                   output_file_path="./data/output/embed-evidence.json")
    
    # retrieve info
    print("Start retrieve info...")
    time.sleep(3)
    print('\033[1A', end='\x1b[2K')
    retrieve_output = biencoder.retrieve(biencoder_valid_train_dataset,
                                         embed_evid_data=embed_evid_data,
                                         k=5,
                                         batch_size =128,
                                         predict_output_path="./data/output/retrieve-train-claims.json")
    
    retrieve_output = biencoder.retrieve(biencoder_valid_dev_dataset,
                                         embed_evid_data=embed_evid_data,
                                         k=5,
                                         batch_size =128,
                                         predict_output_path="./data/output/retrieve-dev-claims.json")
    
if __name__ == "__main__":
    run() 
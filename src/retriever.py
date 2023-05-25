import json
from transformers import AutoModel, AutoTokenizer
from data.biencoder_dataset import BiEncoderDataset
from data.evidence_dataset import EvidenceDataset
from model.dpr.biencoder import BiEncoder
from model.dpr.biencoder_trainer import BiEncoderTrainer

def run():
    
    # create tokenizer
    print("Create tokenizer...")
    simcse_roberta_tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-base",
                                                             use_fast=False)
    print("Tokenizer created.")

    
    # Prepare dataset
    # Training dataset
    print("Load datasets...")
    biencoder_train_dataset = BiEncoderDataset(claim_file_path="./data/train-dev-claims.json",
                                               data_type="train",
                                               evidence_file_path="./data/evidence.json",
                                               tokenizer=simcse_roberta_tokenizer,
                                               lower_case=True,
                                               max_padding_length=128,
                                               evidence_num=8,
                                               rand_seed=666)
    # Testing dataset
    biencoder_valid_train_dataset = BiEncoderDataset(claim_file_path="./data/train-claims.json",
                                                     data_type="predict",
                                                     evidence_file_path="./data/evidence.json",
                                                     tokenizer=simcse_roberta_tokenizer,
                                                     lower_case=True,
                                                     max_padding_length=128,
                                                     evidence_num=0,
                                                     rand_seed=666)

    biencoder_valid_dev_dataset = BiEncoderDataset(claim_file_path="./data/dev-claims.json",
                                                   data_type="predict",
                                                   evidence_file_path="./data/evidence.json",
                                                   tokenizer=simcse_roberta_tokenizer,
                                                   lower_case=True,
                                                   max_padding_length=128,
                                                   evidence_num=0,
                                                   rand_seed=666)
    
    biencoder_test_dataset = BiEncoderDataset(claim_file_path="./data/test-claims-unlabelled.json",
                                                     data_type="predict",
                                                     evidence_file_path="./data/evidence.json",
                                                     tokenizer=simcse_roberta_tokenizer,
                                                     lower_case=True,
                                                     max_padding_length=128,
                                                     evidence_num=0,
                                                     rand_seed=666)
    
    #Evidence dataset
    evidence_dataset = EvidenceDataset(evidence_file_path="./data/evidence.json",
                                       tokenizer=simcse_roberta_tokenizer,
                                       lower_case=True,
                                       max_padding_length=128,
                                       rand_seed=666)
    
    print("Datasets loaded.")
    
    # create model
    print("Create BiEncoder...")
    query_encoder = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-base")
    biencoder = BiEncoder(query_model=query_encoder, similarity_func_type="cosine")
    print("BiEncoder created.")
    
    # train model
    print("Create model trainer.")
    biencoder_trainer = BiEncoderTrainer(model=biencoder,
                                         batch_size=8)
    
    print("Training starts...")
    biencoder_history = biencoder_trainer.train(train_dataset=biencoder_train_dataset,
                                                shuffle=True,
                                                max_epoch=50,
                                                loss_func_type="nll_loss",
                                                optimizer_type="adam",
                                                learning_rate=1e-5)
    
    # Encode evidence
    print("Start embedding evidences....")
    print("Embedding...")
    embed_evid_data = biencoder.get_evidence_embed(evidence_dataset=evidence_dataset,
                                                   batch_size=1000)
    
    # retrieve info
    print("Start retrieve info...")
    retrieve_valid_train_output = biencoder.retrieve(biencoder_valid_train_dataset,
                                                     embed_evid_data=embed_evid_data,
                                                     k=5,
                                                     batch_size =128,
                                                     predict_output_path="./data/output/retrieve-train-claims.json")
    
    retrieve_valit_devoutput = biencoder.retrieve(biencoder_valid_dev_dataset,
                                                  embed_evid_data=embed_evid_data,
                                                  k=5,
                                                  batch_size =128,
                                                  predict_output_path="./data/output/retrieve-dev-claims.json")
    
    retrieve_test_output = biencoder.retrieve(biencoder_test_dataset,
                                              embed_evid_data=embed_evid_data,
                                              k=6,
                                              batch_size =128,
                                              predict_output_path="./data/output/retrieve-test-claims.json")

    
if __name__ == "__main__":
    run() 
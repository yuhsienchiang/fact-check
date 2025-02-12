# Fact Check

This repository contains the code for the final projct of the subject Natural Language Processing COMP90042 2023

## Setup

1. Project dependencies are managed using `poetry`, run poetry command bellow to install dependencies:

```bash
poetry install
```

2. All parameters for the models, training, and predicting are stored in config files at `src/config/`
3. You can run the models by:
   - executing `src/retriever.py` and `src/classify.py` files
   ```bash
   python src/retriever.py
   python src/classify.py
   ```
   - go through the notebooks in `notebooks/` (uncompleted)

## References

- [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)
- [Deep Learning in Information Retrieval. Part II: Dense Retrieval](https://medium.com/@aikho/deep-learning-in-information-retrieval-part-ii-dense-retrieval-1f9fecb47de9)
- [A friendly introduction to Siamese Networks](https://towardsdatascience.com/a-friendly-introduction-to-siamese-networks-85ab17522942)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [Pre-training Methods in Information Retrieval](https://arxiv.org/abs/2111.13853)
- [SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/abs/2104.08821)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)


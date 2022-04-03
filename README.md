# humor-detection
A natural language processing project using BERT to detect humor, loosely replicating this work: https://github.com/Moradnejad/ColBERT-Using-BERT-Sentence-Embedding-for-Humor-Detection

### Data files:
`dataset.csv` - 200k documents, evenly split between documents classified as humorous and nonhumorous
`clean_dataset.csv` - same dataset as above, after processing: expand contractions, pad punctuation with spaces, and handle special characters

### Files for tokenizing:
`bert-embeddings.py` - functions to tokenize the data, relatively faithful to authors' functions (split documents into 5 sentences and tokenize each + tokenize entire document)
`bertembeddings-full.py` - functions to tokenize the documents split into 5 sentences but not tokenizing the entire document
`bertembeddings-simple.py` - tokenize the whole document but not individual MAX_SENTENCES

### Models:
`colbertmodel.py` - model faithful to authors' proposed model (3 parallel layers/sentence * 5 sentences + 3 parallel layers/document = 18 layers)
`replicate_colbert.ipynb` - pmuch Meer's notebook, Jenna shouldn't edit unless she wants to fix merge conflicts as a punishment
`models.ipynb` - have all different models in same place (so far just baseline in there)

### Misc code:
`class-DataGenerator.py` - tokenizing and modeling data in batches to reduce RAM usage

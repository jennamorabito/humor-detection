# The ColBERT Report
A natural language processing project using BERT to detect humor, testing the efficacy of this work: https://github.com/Moradnejad/ColBERT-Using-BERT-Sentence-Embedding-for-Humor-Detection

## Repository structure
Professors will want to check out `models.ipynb` in the Code folder

. \
├── README.md \
├── .gitignore \
├── .DS_Store \
├── Final Report.pdf \
├── Analysis \
    ├── Graphs \
        ├── f1-diff-comparison.png \
        ├── runtime-vs-f1.png \
        ├── training-accuracy.png \
        ├── training-loss.png \
        ├── validation-accuracy.png \
        ├── validation-loss.png \
    ├── Qualitative \
        ├── baseline_misclassified.csv \
        ├── colbert_10k_misclassified.csv \
        ├── colbert_simple_misclassified.csv \
        ├── confusion-matrices.md \
    ├── Quantitative \
        ├── model-results-prettynames.csv \
        ├── model-results.csv \
        ├── train-val-accuracy-loss-all.txt \
        ├── train-val-accuracy-loss-subset-numnames.txt \
        ├── train-val-accuracy-loss-subset.txt \
    ├── .DS_Store \
    ├── 3-models-summary.txt \
    ├── all-models-summary-cheatsheet.png \
├── Code \
    ├── bertembeddings.py \
    ├── class-DataGenerator.py \
    ├── colbertmodel.py \
    ├── models.ipynb \
    ├── replicate_colbert.ipynb \
├── Data \
    ├── clean_dataset.csv \
    ├── dataset.csv \
    ├── test_inputs_1k.npz \
    ├── train_inputs_10k.npz \
    ├── val_inputs_3k.npz

## About our files

### Analysis data files:
/Qualitative is a look at what statements a couple different models misclassified, and /Quantitative has runtimes, F1 scores, etc across models. /Graphs contains a few visualizations about these things.

### Code:
- `bertembeddings.py` - everything we need to tokenize the data, relatively faithful to authors' functions (split documents into 5 sentences and tokenize each + tokenize entire document)
- `class-DataGenerator.py` - tokenizing and modeling data in batches to reduce RAM usage and save the embeddings as .npz files
- `colbertmodel.py` - replicate the ColBERT model
- `models.ipynb` - run models with different combinations of inputs (whole documents, specific sentences, etc)
- `replicate_colbert.ipynb` - data cleaning and try running our first model

### Data files:
- `dataset.csv` - 200k documents, evenly split between documents classified as humorous and nonhumorous
- `clean_dataset.csv` - same dataset as above, after processing: expand contractions, pad punctuation with spaces, and handle special characters
- `*.npz` our training, validation, and test sets of tokenized data, ready to be loaded into a BERT model

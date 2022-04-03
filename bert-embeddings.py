import io
import pandas as pd
import torch
import numpy as np
import nltk
nltk.download('punkt')
# import re
# import os
# import random
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm # for nice progress meters

# read data in as a dataframe
data = pd.read_csv('clean_dataset.csv')

# split into train and test data
x_train, x_test = data['text'][:160000], data['text'][160000:]
y_train, y_test = data['humor'][:160000].values, data['humor'][160000:].values

# cast back into dataframes
x_train, x_test = x_train.to_frame('text'), x_test.to_frame('text')
#y_train, y_test = y_train.to_frame('humor'), y_test.to_frame('humor')

# export tokenized data to load into model later
def save_npz(filename, arr):
    """
    arr: list of 2D arrays
    """
    if '.npz' not in filename:
        filename += '.npz'
    arr_dict = dict(zip(map(str, range(len(arr))), arr))
    np.savez_compressed(filename, **arr_dict)

def load_npz(filename):
    if '.npz' not in filename:
        filename += '.npz'
    return np.load(filename, allow_pickle=True)

# the presets in the author's code
MAX_SENTENCES = 5
MAX_SENTENCE_LENGTH = 20
MAX_LENGTH = 100

# the presets in the author's code
MAX_SENTENCES = 5
MAX_SENTENCE_LENGTH = 20
MAX_LENGTH = 100

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def return_id(str1, truncation_strategy, length):
    """
    helper function to get output from the tokenizer into the right form for the model
    inputs:
    str1, str2: split sentences from the same document
    truncation_strategry: how the BERT encoder should truncate text. default 'longest_first', other options are . . .
    length: either the max sentence length (how many words a sentence can have) or the maximum number of tokens . . . ?

    outputs:
    the three outputs from the BERT encoder for one sentence in a list
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', output_hidden_states=True)
    inputs = tokenizer.encode_plus(str1,
        add_special_tokens=True,
        max_length=length,
        truncation_strategy=truncation_strategy,
        pad_to_max_length = True)

    return [inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]]


def compute_input_arrays(df, columns, tokenizer):
    """
        inputs:
        df: dataframe of text data, needs to have sentences to be run through the model + category labels
        columns: column from df you want to analyze
        tokenizer: BERT tokenizer to use, default=BertTokenizer.from_pretrained('bert-base-uncased')

        outputs: a list of word encodings as np arrays, ready to feed into a neural net
    """

    # create a list populated with empty lists for
    # the maximum number of sentences you think one of your entries could have,
    # times 3 since each sentence has three outputs after being run through the BERT tokenizer in return_id()
    # plus 3 for the three empty lists that the encoding for the whole sentence will need
    model_input = []
    for xx in range((MAX_SENTENCES*3)+3):
        model_input.append([])

    for _, row in tqdm(df[columns].iterrows()):
        i = 0

        # sent
        sentences = nltk.sent_tokenize(row.text)
        for xx in range(MAX_SENTENCES):
            # process 1 sent at a time
            s = sentences[xx] if xx<len(sentences) else ''
            ids_q, masks_q, segments_q = return_id(s, 'longest_first', MAX_SENTENCE_LENGTH)
            model_input[i].append(ids_q)
            i+=1
            model_input[i].append(masks_q)
            i+=1
            model_input[i].append(segments_q)
            i+=1

        # full row
        # outputs from encoder for whole sentence, put into next three empty lists following the encodings for each sentence
        ids_q, masks_q, segments_q = return_id(row.text, 'longest_first', MAX_LENGTH)
        model_input[i].append(ids_q)
        i+=1
        model_input[i].append(masks_q)
        i+=1
        model_input[i].append(segments_q)

    # cast each list as an np array, now we have a list of arrays
    for xx in range((MAX_SENTENCES*3)+3):
        model_input[xx] = np.asarray(model_input[xx], dtype=np.int32)

    print(model_input[0].shape)
    return model_input

inputs_full_50k     = compute_input_arrays(x_train[:50000], ['text'], tokenizer)
save_npz('inputs_full_50k', inputs_full_50k )

test_inputs_full_5k = compute_input_arrays(x_test[:5000], ['text'], tokenizer)
save_npz('test_inputs_full_5k', test_inputs_full_5k)

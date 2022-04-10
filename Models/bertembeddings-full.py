import io
import pandas as pd
# import torch
import numpy as np
import nltk
nltk.download('punkt')
# import re
# import os
# import random
from transformers import BertTokenizer, TFBertModel
# from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
# from sklearn.metrics import f1_score
# from sklearn.model_selection import train_test_split
from tqdm import tqdm # for nice progress meters

# the presets in the author's code
MAX_SENTENCES = 5
MAX_SENTENCE_LENGTH = 20
MAX_LENGTH = 100

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def return_id(str1, truncation_strategy, length):
    """
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
        truncation_strategy=truncation_strategy)

    padding_length = length - len(inputs["input_ids"])
    padding_id = tokenizer.pad_token_id
    input_ids = np.pad(inputs["input_ids"], (0, padding_length), constant_values=padding_id)
    input_masks = np.pad(inputs["attention_mask"], (0, padding_length), constant_values=padding_id)
    input_segments = np.pad(inputs["token_type_ids"], (0, padding_length), constant_values=padding_id)

    return [input_ids, input_masks, input_segments]


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
    model_input = [[]] * ((MAX_SENTENCES+1)*3) # total # of inputs = (# of sentences + doc) * 3 different inputs for bert)

    # padded input ids, masks, and token type ids for texts with less than MAX number of sentences
    empty_ids, empty_masks, empty_segments = return_id('', 'longest_first', MAX_SENTENCE_LENGTH)
    for _, row in tqdm(df[columns].iterrows()):
        i = 0

        # sent
        sentences = nltk.sent_tokenize(row.text) # split each entry into its constituent sentences
        for xx in range(MAX_SENTENCES):
            # process 1 sent at a time
            if xx < len(sentences):
                ids_q, masks_q, segments_q = return_id(sentences[xx], 'longest_first', MAX_SENTENCE_LENGTH) # grab outputs from encoder
            else:
                ids_q = empty_ids
                masks_q = empty_masks
                segments_q = empty_segments
            # put each output for each sentence into sequential spots in the list of empty lists
            model_input[i].append(ids_q)
            i+=1
            model_input[i].append(masks_q)
            i+=1
            model_input[i].append(segments_q)
            i+=1

        # full row
        # outputs from encoder for whole sentence, put into next three empty lists following the encodings for each sentence
        #ids_q, masks_q, segments_q = return_id(row.text, 'longest_first', MAX_LENGTH)
        #model_input[i].append(ids_q)
        i+=1
        #model_input[i].append(masks_q)
        i+=1
        #model_input[i].append(segments_q)

    # cast each list as an np array, now we have a list of arrays
    for xx in range((MAX_SENTENCES*3)+3):
        model_input[xx] = np.asarray(model_input[xx], dtype=np.int32)

    print(model_input[0].shape)
    return model_input

# read in data
data = pd.read_csv('clean_dataset.csv')

# split into train and test data
x_train, x_test = data['text'][:160000], data['text'][160000:]
y_train, y_test = data['humor'][:160000].values, data['humor'][160000:].values

# cast back into dataframes
x_train, x_test = x_train.to_frame('text'), x_test.to_frame('text')
#y_train, y_test = y_train.to_frame('humor'), y_test.to_frame('humor')


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
inputs      = compute_input_arrays(x_train[:15000], ['text'], tokenizer)
test_inputs = compute_input_arrays(x_test[:5000], ['text'], tokenizer)

np.save('inputs-sentences-15k.npy', inputs)
np.save('test-inputs-sentences-5k.npy', test_inputs)

# inputs      = compute_input_arrays(x_train, ['text'], tokenizer)
# test_inputs = compute_input_arrays(x_test, ['text'], tokenizer)

# print(len(inputs), len(inputs[0]), len(inputs[0][0]))

# # check out input for 7th row
# xx = 7
# print(x_train.iloc[xx,0])
# print(nltk.sent_tokenize(x_train.iloc[xx,0]))
# inputs[0][xx], inputs[3][xx], inputs[6][xx], inputs[15][xx]

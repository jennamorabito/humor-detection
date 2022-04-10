import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

# import tensorflow_hub as hub
import tensorflow as tf
# import bert_tokenization as tokenization
import tensorflow.keras.backend as K
from tensorflow import keras 

import os
from scipy.stats import spearmanr
from math import floor, ceil
# from transformers import *
from transformers import TFBertModel, BertTokenizer

import seaborn as sns
import string
import re    #for regex

# import keras model layers
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Concatenate

SENT_INPUT_LEN = 20
DOC_INPUT_LEN = 100
DROPOUT_RATE = 0.2

# 18 inputs, 3 for each parallel path (5 sentence-level paths & 1 document-level path)
input_sent1_1 = Input(shape=(SENT_INPUT_LEN,), name='input_ii_sent1') # input IDs
input_sent1_2 = Input(shape=(SENT_INPUT_LEN,), name='input_am_sent1') # attention masks
input_sent1_3 = Input(shape=(SENT_INPUT_LEN,), name='input_tti_sent1') # token type IDs

input_sent2_1 = Input(shape=(SENT_INPUT_LEN,), name='input_ii_sent2')
input_sent2_2 = Input(shape=(SENT_INPUT_LEN,), name='input_am_sent2')
input_sent2_3 = Input(shape=(SENT_INPUT_LEN,), name='input_tti_sent2')

input_sent3_1 = Input(shape=(SENT_INPUT_LEN,), name='input_ii_sent3')
input_sent3_2 = Input(shape=(SENT_INPUT_LEN,), name='input_am_sent3')
input_sent3_3 = Input(shape=(SENT_INPUT_LEN,), name='input_tti_sent3')

input_sent4_1 = Input(shape=(SENT_INPUT_LEN,), name='input_ii_sent4')
input_sent4_2 = Input(shape=(SENT_INPUT_LEN,), name='input_am_sent4')
input_sent4_3 = Input(shape=(SENT_INPUT_LEN,), name='input_tti_sent4')

input_sent5_1 = Input(shape=(SENT_INPUT_LEN,), name='input_ii_sent5')
input_sent5_2 = Input(shape=(SENT_INPUT_LEN,), name='input_am_sent5')
input_sent5_3 = Input(shape=(SENT_INPUT_LEN,), name='input_tti_sent5')

input_doc_1 = Input(shape=(DOC_INPUT_LEN,), name='input_ii_doc')
input_doc_2 = Input(shape=(DOC_INPUT_LEN,), name='input_am_doc')
input_doc_3 = Input(shape=(DOC_INPUT_LEN,), name='input_tti_doc')

# embedding layer for sentences and documents
#bert_embeddings = Embedding(num_tokens,embedding_dim,embeddings_initializer=keras.initializers.Constant(embedding_matrix),trainable=False)
bert_model = TFBertModel.from_pretrained('bert-base-uncased')
bert_embeddings1 = bert_model(input_ids=input_sent1_1, attention_mask=input_sent1_2, token_type_ids=input_sent1_3)
bert_embeddings2 = bert_model(input_ids=input_sent2_1, attention_mask=input_sent2_2, token_type_ids=input_sent2_3)
bert_embeddings3 = bert_model(input_ids=input_sent3_1, attention_mask=input_sent3_2, token_type_ids=input_sent3_3)
bert_embeddings4 = bert_model(input_ids=input_sent4_1, attention_mask=input_sent4_2, token_type_ids=input_sent4_3)
bert_embeddings5 = bert_model(input_ids=input_sent5_1, attention_mask=input_sent5_2, token_type_ids=input_sent5_3)
bert_embeddings6 = bert_model(input_ids=input_sent6_1, attention_mask=input_sent6_2, token_type_ids=input_sent6_3)

# get pooled vectors of BERT sentence embeddings
x1 = bert_embeddings1[1] 
x2 = bert_embeddings2[1] 
x3 = bert_embeddings3[1]
x4 = bert_embeddings4[1]
x5 = bert_embeddings5[1]
x6 = bert_embeddings6[1] 

# fully connected layer w/ dropout
h1_1 = Dense(32, activation='relu', name="hidden1_sent1")(x1)
h1_2 = Dense(32, activation='relu', name="hidden1_sent2")(x2)
h1_3 = Dense(32, activation='relu', name="hidden1_sent3")(x3)
h1_4 = Dense(32, activation='relu', name="hidden1_sent4")(x4)
h1_5 = Dense(32, activation='relu', name="hidden1_sent5")(x5)
h1_6 = Dense(256, activation='relu', name="hidden1_doc")(x6)

h1_dropout1 = Dropout(DROPOUT_RATE, name="h1_dropout_sent1")(h1_1) ####################################################
h1_dropout2 = Dropout(DROPOUT_RATE, name="h1_dropout_sent2")(h1_2) ####################################################
h1_dropout3 = Dropout(DROPOUT_RATE, name="h1_dropout_sent3")(h1_3) #                rate TO BE CHANGED                # 
h1_dropout4 = Dropout(DROPOUT_RATE, name="h1_dropout_sent4")(h1_4) #                                                  #
h1_dropout5 = Dropout(DROPOUT_RATE, name="h1_dropout_sent5")(h1_5) ####################################################
h1_dropout6 = Dropout(DROPOUT_RATE, name="h1_dropout_doc")(h1_6)   ####################################################

# fully connected layer
h2_1 = Dense(8, activation='relu', name="hidden2_sent1")(h1_dropout1)
h2_2 = Dense(8, activation='relu', name="hidden2_sent2")(h1_dropout2)
h2_3 = Dense(8, activation='relu', name="hidden2_sent3")(h1_dropout3)
h2_4 = Dense(8, activation='relu', name="hidden2_sent4")(h1_dropout4)
h2_5 = Dense(8, activation='relu', name="hidden2_sent5")(h1_dropout5)
h2_6 = Dense(64, activation='relu', name="hidden2_doc")(h1_dropout6)

# concatenate outputs of all 6 parallel layers
xx = Concatenate()([h2_1, h2_2, h2_3, h2_4, h2_5, h2_6])

# fully connected layer w/ dropout for concatenated inputs
h3 = Dense(512, activation='relu', name="hidden3")(xx)
h3_dropout = Dropout(DROPOUT_RATE)(h3) ################ rate TO BE CHANGED ################

# fully connected layer
h4 = Dense(256, activation='relu', name="hidden4")(h3_dropout)

# final output layer
yhat = Dense(1, activation='sigmoid', name="output")(h4) # need to figure out dropout rate
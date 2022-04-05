###########################################################################
### CREDIT: Adapted from Issa Annamoradnejad and Gohar Zoghi's code for ###
### humor detection. Paper: https://arxiv.org/pdf/2004.12765v5.pdf      ###
###########################################################################

import io
import numpy as np
import nltk
import transformers
from transformers import BertTokenizer, TFBertModel
from tqdm import tqdm # for nice progress meters
nltk.download('punkt')

# the presets in the author's code
MAX_SENTENCES = 5
MAX_SENTENCE_LENGTH = 20
MAX_LENGTH = 100

def return_id(str1, truncation_strategy, length):
    """
    helper function to get output from the tokenizer into the right form for the model
    inputs:
    str1: split sentences from the same document
    truncation_strategry: how the BERT encoder should truncate text. default 'longest_first', other options are . . .
    length: either the max sentence length (how many words a sentence can have) or the maximum number of tokens . . . ?

    outputs:
    the three outputs from the BERT encoder for one sentence in a list
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', output_hidden_states=True)
    inputs = tokenizer.encode_plus(str1,
        add_special_tokens=True,
        max_length=length,
        truncation_strategy='longest_first',
        pad_to_max_length = True)

    return [inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]]


def compute_input_arrays(df, columns, tokenizer):
    model_input = []
    for xx in range((MAX_SENTENCES*3)+3):
        model_input.append([])

    for _, row in tqdm(df[columns].iterrows()):
        i = 0

        # sent
        sentences = nltk.sent_tokenize(row.text)
        for xx in range(MAX_SENTENCES):
            s = sentences[xx] if xx<len(sentences) else ''
            ids_q, masks_q, segments_q = return_id(s, 'longest_first', MAX_SENTENCE_LENGTH)
            model_input[i].append(ids_q)
            i+=1
            model_input[i].append(masks_q)
            i+=1
            model_input[i].append(segments_q)
            i+=1

        # full row
        ids_q, masks_q, segments_q = return_id(row.text, 'longest_first', MAX_LENGTH)
        model_input[i].append(ids_q)
        i+=1
        model_input[i].append(masks_q)
        i+=1
        model_input[i].append(segments_q)

    for xx in range((MAX_SENTENCES*3)+3):
        model_input[xx] = np.asarray(model_input[xx], dtype=np.int32)

    print(model_input[0].shape)
    return model_input

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 14:31:14 2021

@author: Sagun Shakya
"""

# Importing necessary libraries.
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
np.random.seed(0)
plt.style.use("ggplot")

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from torch.utils.data import DataLoader

#from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
#from keras.preprocessing.sequence import pad_sequences

# LOCAL MODULES.
import utils
from BiLSTM_module import BilSTM_model
from Dataset_module import NERDataset

# Filter out warning messages.
from warnings import filterwarnings
filterwarnings('ignore')

# Parsing data.
data_path = r'D:\ML_projects\NER_pytorch_english\data'
filename = os.path.join(data_path, 'ner_dataset.csv')

data = pd.read_csv(filename, encoding="latin1")
data = data.fillna(method="ffill")

# Creating a list of unique words and tags.
words = data['Word'].unique().tolist()
words = words + ['<UNK>', '<PAD>', 'ENDTAG']
tags = data['Tag'].unique().tolist()

# Global constants.
TEST_SIZE = 0.2
RANDOM_STATE = 1

# For NN.
VOCAB_SIZE = len(words)
MAX_LEN = 50
EMBEDDING_DIM = 50
HIDDEN_DIM = 100
DROPOUT_PROB = 0.1
BATCH_SIZE = 8
EPOCHS = 3
num_tags = len(tags)

# Aggregation function to make a list of tokens/tags in each row.
agg_function_word_only = lambda df: [w for w in df['Word'].values.tolist()]
agg_function_tag_only = lambda df: [t for t in df['Tag'].values.tolist()]

grouped = data.groupby("Sentence #")

# List of word_lists and tag_tokens.
sentences = grouped.apply(agg_function_word_only).tolist()
tag_sequence = grouped.apply(agg_function_tag_only).tolist() 

# Mappings from word/tag to its IDs.
word2idx = {word : ii for ii, word in enumerate(words, 1)}
tag2idx = {tag : ii for ii, tag in enumerate(tags, 0)}
tag2idx['<pad>'] = 17

# Inverse mapping.
idx2word = {ii : word for ii, word in enumerate(words, 1)}
idx2tag = {ii : tag for ii, tag in tag2idx.items()}
idx2tag[17] = '<pad>'

# ID for <PAD> in a sentence.
pad_id = word2idx['<PAD>']

# List of token IDs for wach sentence and the tagset. 
unpadded_sequences = [utils.word_list2id_list(SEQ, word2idx, tag2idx, 'word') for SEQ in sentences]
unpadded_sequences = list(map(torch.tensor, unpadded_sequences))

unpadded_tags = [utils.word_list2id_list(SEQ, word2idx, tag2idx, 'tag') for SEQ in tag_sequence]
unpadded_tags = list(map(torch.tensor, unpadded_tags))

# Padding the sentences with the ID for <PAD> and tags with the ID for <pad>.
# Maximum length for padding is 50.
# If length of a tensor exceeds 50, it'll be post-truncated.
X = [F.pad(tensor_i, pad = (0, MAX_LEN - len(tensor_i)), mode = "constant", value = pad_id) for tensor_i in unpadded_sequences]
Y = [F.pad(tensor_i, pad = (0, MAX_LEN - len(tensor_i)), mode = "constant", value = tag2idx['<pad>']) for tensor_i in unpadded_tags]

# Train-Test split in the ratio 4:1.
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# Creating training and testing iterators.
train_loader = DataLoader(NERDataset(x_train, y_train), batch_size = BATCH_SIZE, shuffle=True)
test_loader = DataLoader(NERDataset(x_test, y_test), batch_size = BATCH_SIZE, shuffle=True)

# Model Architecture.
model = BilSTM_model(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, DROPOUT_PROB, MAX_LEN, num_tags, BATCH_SIZE)

    
# Training Phase.

# Defining the parameters of the network.
optimizer = Adam(model.parameters(), lr = 0.001)

# Start  training.
for ee in range(EPOCHS):
    training_loss_per_epoch = 0
    training_accuracy_per_epoch = 0

    val_loss_per_epoch = 0
    val_accuracy_per_epoch = 0

    for ii, ((sample, seq_len), tag) in enumerate(train_loader):
        # Clear the gradients.
        model.zero_grad()

        # Generating the output of shape (batch_size, num_tags, max_len).
        preds = model.forward(sample, seq_len)
        if preds is None:
            print("Skipped in batch: ", ii)
            continue        

        # Using negative log-likelihood loss function. 
        # If CrossEntropyLoss is used, there's no need to apply softmax. Can take input directly from the last linear layer.
        # Preventing the <pad> element from contributing to the loss.
        loss = F.nll_loss(preds, tag, ignore_index = tag2idx['<pad>'])
        loss_rounded = round(loss.item(), 4)
        print(f"\nLoss value in this iteration (Epoch : {ee} & Batch : {ii}): {loss_rounded}")

        # Backpropagation.
        loss.backward()

        # Update the weights.
        optimizer.step()

        # Categorical Accuracy.
        train_acc_per_batch = utils.categorical_accuracy(preds, tag, tag_pad_value = tag2idx['<pad>'])
        print(f"Accuracy in this iteration (Epoch : {ee} & Batch : {ii}): {train_acc_per_batch}")

        ##break

    break


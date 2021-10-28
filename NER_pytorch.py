# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 14:31:14 2021

@author: Sagun Shakya
"""

# Importing necessary libraries.
import matplotlib.pyplot as plt
plt.style.use("ggplot")

from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

# LOCAL MODULES.
from BiLSTM_module import BilSTM_model
from Dataset_module import NERDataset
from EarlyStopping_module import EarlyStopping
from Load_dataset import LoadDataset
from dynamic_model import DynamicModel

# Filter out warning messages.
from warnings import filterwarnings
filterwarnings('ignore')

#################### CONSTANTS ####################
# Global constants.
TEST_SIZE = 0.2
RANDOM_STATE = 1

# For NN.
MAX_LEN = 50
EMBEDDING_DIM = 50
HIDDEN_DIM = 100
DROPOUT_PROB = 0.1
BATCH_SIZE = 8
EPOCHS = 3
PATIENCE = 5
#-------------------------------------------------#

#################### VARIABLE INITIALIZATION ####################
# Root and filename.
data_path = r'D:\ML_projects\NER_pytorch_english\data'
filename = 'ner_dataset.csv'

# Main Dataholder class.
main = LoadDataset(data_path, filename)

# Unique words.
words = main.words

# Size of the vocabulary.
VOCAB_SIZE = len(words)

# Unique tags.
tags = main.tags

# Number of output classes.
num_tags = len(tags)

# Sentences. List of word_lists.
sentences = main.sentences

# Tag Sequence.
tag_sequence = main.tag_sequence

# Seq2idx.
word2idx = main.word2idx
tag2idx = main.tag2idx

# Idx2seq.
idx2word = main.idx2word
idx2tag = main.idx2tag

# Pad_id.
pad_id = main.word_pad_id
tag_pad_id = main.tag_pad_id

# Unpadded sequences and tags.
unpadded_sequences = main.unpadded_sequences
unpadded_sequences = list(map(torch.tensor, unpadded_sequences))

unpadded_tags = main.unpadded_tags
unpadded_tags = list(map(torch.tensor, unpadded_tags))

# Padding the sentences with the ID for <PAD> and tags with the ID for <pad>.
# Maximum length for padding is 50.
# If length of a tensor exceeds 50, it'll be post-truncated.
X = [F.pad(tensor_i, pad = (0, MAX_LEN - len(tensor_i)), mode = "constant", value = pad_id) for tensor_i in unpadded_sequences]
Y = [F.pad(tensor_i, pad = (0, MAX_LEN - len(tensor_i)), mode = "constant", value = tag_pad_id) for tensor_i in unpadded_tags]

# Train-Test split in the ratio 4:1.
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# Creating training and testing iterators.
train_loader = DataLoader(NERDataset(x_train, y_train, pad_id), batch_size = BATCH_SIZE, shuffle=True)
test_loader = DataLoader(NERDataset(x_test, y_test, pad_id), batch_size = BATCH_SIZE, shuffle=True)
#---------------------------------------------------------------#

#################### MODEL BUILDING ####################
# Model Architecture.
model = BilSTM_model(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, DROPOUT_PROB, MAX_LEN, num_tags, BATCH_SIZE)

# Defining the parameters of the network.
optimizer = Adam(model.parameters(), lr = 0.001)
 
# Defining early stopping object.
early_stopping = EarlyStopping(patience = PATIENCE, verbose = True, delta = 0.0001)

# Training Phase.
dynamic_model = DynamicModel(model, optimizer, tag_pad_id)

history = dynamic_model.fit(train_loader, 
                            test_loader, 
                            epochs = EPOCHS, 
                            n = 50, 
                            early_stopping_callback = early_stopping, 
                            return_cache = True, 
                            plot_history = True)


 #---------------------------------------------------------------#   
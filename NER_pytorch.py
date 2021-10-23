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
from EarlyStopping_module import EarlyStopping

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
PATIENCE = 5
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
train_loader = DataLoader(NERDataset(x_train, y_train, pad_id), batch_size = BATCH_SIZE, shuffle=True)
test_loader = DataLoader(NERDataset(x_test, y_test, pad_id), batch_size = BATCH_SIZE, shuffle=True)

# Model Architecture.
model = BilSTM_model(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, DROPOUT_PROB, MAX_LEN, num_tags, BATCH_SIZE)

    
# Training Phase.

# Cache for accuracy and losses in each epoch for training and validatin sets.
accuracy_cache_train = {"epoch_" + str(ii) : [] for ii in range(EPOCHS)}
accuracy_cache_val = {"epoch_" + str(ii) : [] for ii in range(EPOCHS)}

loss_cache_train = {"epoch_" + str(ii) : [] for ii in range(EPOCHS)}
loss_cache_val= {"epoch_" + str(ii) : [] for ii in range(EPOCHS)}

# Defining the parameters of the network.
optimizer = Adam(model.parameters(), lr = 0.001)

# Defining early stopping object.
early_stopping = EarlyStopping(patience = PATIENCE, verbose = True, delta = 0.0001)

# Start  training.
for ee in range(EPOCHS):
    
    # Empty lists for storing the train/validation accuracy and losses for each iteration in the ee-th epoch.
    # These will be stored in the dictionary containing the cache for each epoch.
    
    accuracy_train = []
    accuracy_val = []
    
    loss_train = []
    loss_val = []
    
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
        #print(f"\nLoss value in this iteration (Epoch : {ee} & Batch : {ii}): {loss_rounded}")

        # Backpropagation.
        loss.backward()

        # Update the weights.
        optimizer.step()

        # Categorical Accuracy.
        train_acc_per_iter = utils.categorical_accuracy(preds, tag, tag_pad_value = tag2idx['<pad>'])
        #print(f"Accuracy in this iteration (Epoch : {ee} & Batch : {ii}): {train_acc_per_iter}")

        # Calculate the loss and accuracy for the validation set in every 50 iteration.
        if (ii + 1) % 50 == 0:
            avg_train_accuracy = []
            avg_val_accuracy = []
            
            avg_train_loss = []
            avg_val_loss = []
            
            for (sample_t, seq_len_t), tag_t in test_loader:
                
                # Forward Propagation.
                preds_val = model.forward(sample_t, seq_len_t)
                
                # Calculate the loss.
                val_loss_per_batch = F.nll_loss(preds_val, tag_t, ignore_index = tag2idx['<pad>'])
                val_loss_per_batch = val_loss_per_batch.item()
                
                # Calculating the accuracy.
                val_accuracy_per_batch = utils.categorical_accuracy(preds_val, tag_t, tag_pad_value = tag2idx['<pad>'])
                
                # Storing the losses and accuracies for validation batches (NOT THE TRAINING BATCH) in this iteration.
                ## Validation Set.
                avg_val_accuracy.append(val_accuracy_per_batch)
                avg_val_loss.append(val_loss_per_batch)
            
            ## Train Set.
            ### Stores the training accuracy and loss for the 50th, 100th, ..., (50*n)th iteration only. 
            avg_train_accuracy.append(train_acc_per_iter)
            avg_train_loss.append(loss_rounded)
            
            # Calculating the average loss for the valdation set in this iteration.
            avg_val_accuracy = utils.compute_average(avg_val_accuracy)
            avg_val_loss = utils.compute_average(avg_val_loss)
            
            # Verbose.
            epoch_step_info = f"Epoch [{ee+1} / {EPOCHS}], Step [{ii+1} / {len(train_loader)}], "
            loss_info = f"Training Loss: {avg_train_loss[0]:.4f}, Validation loss: {avg_val_loss:.4f}, "
            accuracy_info = f"Training Accuracy: {avg_train_accuracy[0]:.4f}, Validation Accuracy: {avg_val_accuracy:.4f}"
            print(epoch_step_info + loss_info + accuracy_info)
            
        #if ii == 200:
            #break
        
    # Storing the cache for this epoch into the dictionaries above.
    
    # Train Set.
    accuracy_cache_train['epoch_' + str(ee)] = avg_train_accuracy
    loss_cache_train['epoch_' + str(ee)] = avg_train_loss
    
    # Validation Set.
    accuracy_cache_val['epoch_' + str(ee)] = avg_val_accuracy
    loss_cache_val['epoch_' + str(ee)] = avg_val_loss
    
    early_stopping(avg_val_loss, model)
    if early_stopping.early_stop:
        print("Early Stopping in Epoch ", ee)
        break
    
    break
    
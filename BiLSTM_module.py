# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 09:12:26 2021

@author: Sagun Shakya
"""
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BilSTM_model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout_prob, max_len, num_tags, batch_size):
        super(BilSTM_model, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        self.max_len = max_len    
        self.num_tags = num_tags
        self.batch_size = batch_size

        self.word_embeddings = nn.Embedding(num_embeddings = self.vocab_size, embedding_dim = self.embedding_dim)

        #### TO USE A SET OF PRE-TRAINED EMBEDDINGS. ####
        # self.word_embeddings.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        # self.word_embeddings.weight.requires_grad = False

        self.dropout_emb = nn.Dropout(self.dropout_prob)
        self.bilstm = nn.LSTM(input_size = self.embedding_dim, 
                              hidden_size = self.hidden_dim,
                              dropout = 0.1, 
                              bias = True,  
                              bidirectional = True, 
                              batch_first = True)

        self.linear = nn.Linear(self.hidden_dim*2, self.num_tags)

    def forward(self, sentence, seq_len):

        # Length of each sequence.
        #seq_len = torch.LongTensor(list(map(len, sentence)))
        
        # Embedding layer.
        sent_embeddings = self.word_embeddings(sentence)

        # Adding dropout to the embedding output.
        sent_embeddings = self.dropout_emb(sent_embeddings)

        # Packing the output of the embedding layer.
        packed_input = pack_padded_sequence(sent_embeddings, 
                                            lengths = seq_len.clamp(max = 192), 
                                            batch_first = True, 
                                            enforce_sorted = False)
        
        diff = packed_input.batch_sizes.sum().item() - packed_input.data.shape[0]
        if diff > 0:
            print("\nShape mismatch found.")
            print("Sum of seq_len: ", packed_input.batch_sizes.sum().item())
            print("Shape of packed input: ", packed_input.data.shape)
            print("Skipping...")
            return None
        
        # BiLSTM layer.
        packed_output, (h_t, c_t) = self.bilstm(packed_input)

        # Inverting the packing operation.
        out, input_sizes = pad_packed_sequence(packed_output, batch_first = True, total_length = self.max_len)

        #Linear layer.
        output = self.linear(out)

        # Softmax. Then, logarithmic transform.
        pred_prob = F.log_softmax(output, dim=1)

        # Transposing the dimension: (batch_size, max_len, num_tags) --> (batch_size, num_tags, max_len).
        pred_prob = pred_prob.permute(0, 2, 1) 

        return pred_prob


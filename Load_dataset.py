# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 09:46:16 2021

@author: Sagun Shakya
"""
import os
import pandas as pd

# Local Modules.
import utils

class LoadDataset:
    def __init__(self, root_path, filename):
        self.root_path = root_path
        self.filename = filename
        self.filename = os.path.join(self.root_path, self.filename)
        
        # Dataframe.
        self.data = pd.read_csv(self.filename, encoding="latin1").fillna(method="ffill")
        
        # Unique Words.
        self.words = self.data['Word'].unique().tolist() + ['<UNK>', '<PAD>', 'ENDTAG']
        
        # Unique Tags.
        self.tags = self.data['Tag'].unique().tolist()
        
        # Grouped data.
        self.grouped = self.data.groupby("Sentence #")
        
        # Aggregation function to make a list of tokens in each row.
        agg_function_word_only = lambda df: [w for w in df['Word'].values.tolist()]
        
        # List of word_lists.
        self.sentences = self.grouped.apply(agg_function_word_only).tolist()
        
        # Aggregation function to make a list of tags in each row.
        agg_function_tag_only = lambda df: [t for t in df['Tag'].values.tolist()]
        
        # List of tags_tokens.
        self.tag_sequence = self.grouped.apply(agg_function_tag_only).tolist()
        
        # Mappings from word/tag to its IDs.
        self.word2idx = {word : ii for ii, word in enumerate(self.words, 1)}
        self.tag2idx = {tag : ii for ii, tag in enumerate(self.tags, 0)}
        self.tag2idx['<pad>'] = 17
        
        # Inverse mapping.
        self.idx2word = {ii : word for ii, word in enumerate(self.words, 1)}
        self.idx2tag = {ii : tag for ii, tag in self.tag2idx.items()}
        self.idx2tag[17] = '<pad>'
        
        # Pad Index for word_sequence and tag_sequence.
        self.word_pad_id = self.word2idx['<PAD>']
        self.tag_pad_id = self.tag2idx['<pad>']
        
        # Unpadded sequences.
        # List of token IDs for wach sentence and the tagset. 
        self.unpadded_sequences = [utils.word_list2id_list(SEQ, self.word2idx, self.tag2idx, mapper='word') for SEQ in self.sentences]
        #self.unpadded_sequences = list(map(torch.tensor, self.unpadded_sequences))
        
        self.unpadded_tags = [utils.word_list2id_list(SEQ, self.word2idx, self.tag2idx, mapper='tag') for SEQ in self.tag_sequence]
        #self.unpadded_tags = list(map(torch.tensor, self.unpadded_tags))
        
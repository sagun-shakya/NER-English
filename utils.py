# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 07:18:59 2021

@author: Sagun Shakya
"""
import torch

def word_list2id_list(word_list, word2idx, tag2idx, mapper = 'word'):
    '''
    For a given list of tokens, the function replaces the tokens with
    their corresponding idx in the vocabulary.
    '''
    try:
        if mapper == 'word':
            return [word2idx[WORD] for WORD in word_list]
        else:
            return [tag2idx[WORD] for WORD in word_list]
    except:
        return []
        
def categorical_accuracy(model_output, true_labels, tag_pad_value = 17):
    try:
        predicted_labels = model_output.argmax(axis = 1)

        error_msg = f'The shape of the predicted_labels doesnt match with that of true_labels.'
        error_msg += f'\nShape of predicted_labels: {predicted_labels.shape}'
        error_msg += f'\nShape of true_labels: {true_labels.shape}'
        assert predicted_labels.shape == true_labels.shape, error_msg
        
        # Mask to filter in non-padded elements.
        non_pad_mask = (true_labels != tag_pad_value)

        model_output_smooth = predicted_labels[non_pad_mask]
        true_labels_smooth = true_labels[non_pad_mask]

        assert model_output_smooth.shape == true_labels_smooth.shape, "The shape of the flattened outputs/labels do not match."

        res = model_output_smooth.eq(true_labels_smooth).to(torch.int8)     # Binary value. 1 for match, 0 for no match.
        correct = res.sum()
        total = len(res)                                                    # Sum of Lengths of sequences in the batch.
        accuracy = res.sum()/len(res)
        return round(accuracy.item(), 4)

    except AssertionError as msg:
        print(msg)
        
# Calculating the average loss for the valdation set in this iteration.
compute_average = lambda arr: sum(arr) / len(arr)
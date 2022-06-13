import os
from typing import Optional, List
import torch
import pickle
from collections import Counter
from pandas import DataFrame


class SSTDataset(torch.utils.data.Dataset):
    def __init__(self, data, embedder, input_type):
        """
        Initializes an instance of ourData.
        Inputs:
            data (pandas.DataFrame): The data set to work with.
            embedder (gensim.models.keyedvectors.Word2VecKeyedVectors): Pre-trained word embbeddings.
            input_type (str): Determines what type of data to use.
        """
        self.label = list(data.label)
        self.tokens = list(data.tokens)
        self.lemmatized = list(data.lemmatized)

        self.input_type = input_type
        if self.input_type == "lemmatized":
            self.lemmatized_wout_pos = list(data.lemmatized_wout_pos)

        self.label_vocab = list(set(self.label))
        self.label_indexer = {i: n for n, i in enumerate(self.label_vocab)}

        self.embedder = embedder
        self._unk = embedder.vocab['<unk>'].index


    def __getitem__(self, index):
        """
        A magic function that allows us to use the index operator on our dataset.
        Inputs:
            index (int): Index value to get.
        Outputs:
            X (torch.Tensor): Input variables.
            length (torch.Tensor):
            y (torch.Tensor): Labels.
        """
        if self.input_type == "raw":
            current_text = self.tokens[index]
        elif self.input_type == "pos_tagged":
            current_text = self.lemmatized[index]
        elif self.input_type == "lemmatized":
            current_text = self.lemmatized_wout_pos[index]
        else:
            print("Invalid input type!")

        current_label = self.label[index]
        X = torch.LongTensor([self.embedder.vocab[token].index if token in self.embedder.vocab else self._unk
                              for token in current_text])
        y = self.label_indexer[current_label]
        y = torch.LongTensor([y])
        length = torch.tensor(len(current_text))

        return X,length,y

    def __len__(self):
        """
        Returns the number of rows in the dataset.
        """

        return len(self.tokens)
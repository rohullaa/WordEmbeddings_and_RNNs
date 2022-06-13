import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import logging as log
import time
import nltk


def read_seperate_data(path="data/stanford_sentiment_binary.tsv", sep='\t'):
    """
    Read the data from file.

    Input:
        path (str): Path to the raw data.
    Output:
        df (pandas.DataFrame): The data extracted from the path.
    """

    log.info("Reading the dataset...")

    df = pd.read_csv(path,sep='\t')
    texts = df.tokens.values
    labels = df.label.values

    return df, texts, labels


def find_most_common(texts, num_words, num_most_occ):
    """
    Inputs:
        texts (numpy.array): Texts extracted from the data.
        num_words (int): Number of most common words to find.
        num_most_occ (int): If a word occurs less than this number, that word is not included
                            in the common vocabulary. Hence it is not included as a feature.
    Output:
        most_common_words (numpy.array) Array containing the common vocabulary of the corpus.
    """
    text_vocab = ""

    for i,text in enumerate(texts):
        text = text.split()
        text = [token.lower() for token in text]
        counter = Counter(text)
        most_occurcurences = counter.most_common(num_words)

        for w,t in most_occurcurences:
            if t > num_most_occ: #if the occurences of words is not 1
                text_vocab += w + " "


    text_vocab = np.array(list(set(text_vocab.split())))


    most_common_words = np.unique(text_vocab)

    return most_common_words


def generate_bow(texts,vocabulary):
    """
    Generates the BoW matrix.

    Input:
        texts (numpy.array): Array containing texts we extracted from the dataset.
        vocabulary (list): Common vocabulary containing the unique tokenized words extracted
                           from the dataset.
    Outputs:
        BoW (numpy.array): Multidimensional array containing Bag of Words vectors. Has shape
                           (texts.shape[0] * len(vocabulary)).
    """

    log.info("Generating bag of words...")

    t0 = time.time()
    vectorizer = CountVectorizer(vocabulary = vocabulary)
    BoW = vectorizer.fit_transform(texts)
    BoW = BoW.toarray()

    log.info(f"Time used for generating BoW and common_vocabulary: {time.time()-t0} s")

    return BoW

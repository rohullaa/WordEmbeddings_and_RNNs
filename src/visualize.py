import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import logging as log
import time
import numpy as np
## importing the main.py file
import helping_functions as help
#from nltk import ngrams
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
import unicodedata
import re

from IPython import embed

def basic_clean(text):
  """
  This function is taken from https://towardsdatascience.com/from-dataframe-to-n-grams-e34e29df3460
  A simple function to clean up the data. All the words that
  are not designated as a stop word is then lemmatized after
  encoding and basic regex parsing are performed.
  """
  wnl = nltk.stem.WordNetLemmatizer()
  stopwords = nltk.corpus.stopwords.words('english')
  text = (unicodedata.normalize('NFKD', text)
    .encode('ascii', 'ignore')
    .decode('utf-8', 'ignore')
    .lower())
  words = re.sub(r'[^\w\s]', '', text).split()
  return [wnl.lemmatize(word) for word in words if word not in stopwords]


def visualize_word_count(df, texts, labels, BoW, vocabulary, exclude_first= 10, n_words = 50, show=False):
    """
    FUNCTION USED TO CREATE FIG 1 IN THE REPORT.
    Examines the dataset, finds out which words are used most frequently in data belonging
    to each source.
    """
    unique_labels = np.unique(labels)

    fig = plt.figure(figsize=(12, 12))
    #fig, axs = plt.subplots((len(unique_labels),1))

    for idx, l in enumerate(unique_labels):
        df_label_indices = df[df.label == l].index.copy()
        matrix_label = sum(BoW[df_label_indices])

        dictionary = dict(zip(vocabulary, matrix_label))
        dictionary = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
        dictionary = dict(dictionary)

        first_n = list(dictionary.items())[exclude_first: n_words + exclude_first]
        words = []
        word_freq = []

        for item in first_n:
            words.append(item[0])
            word_freq.append(item[1])
        fig.add_subplot(2, 1, idx+1)
        plt.bar(words, word_freq, color="black")
        plt.xticks(rotation='vertical', fontsize=8)
        plt.title(f"most common {n_words} words in {l} reviews", fontsize=8, fontweight="bold")
        plt.ylabel("Word frequency", fontsize=8, fontweight="bold")
        plt.xlabel("Words", fontsize=8, fontweight="bold")
    fig.tight_layout()

    plt.savefig(f"plots/wordfreq_{n_words}_exclude_{exclude_first}.png")

    if show:
        plt.show()
    else:
        plt.close()

def visualize_ngrams(df, n, most_common):# texts, labels, n, BoW, vocabulary, exclude_first= 10, n_words = 50, show=False):
    df_pos = df[df.label == "positive"].copy()
    df_neg = df[df.label == "negative"].copy()

    fig = plt.figure(figsize=(12, 12))

    for idx, data, label_type in [(0, df_pos,"positive"), (1, df_neg, "negative")]:
        grams = []

        texts = data.tokens.values

        for text in texts:
            words = basic_clean(text)
            #text = [word.lower() for word in text.split()]
            n_gram = nltk.ngrams(words, n)
            for gram in n_gram:
                grams.append(gram)

        #grams = np.asarray(grams, dtype=object)
        #embed()
        res = pd.Series(grams).value_counts()[:most_common]
        fig.add_subplot(2, 1, idx+1)
        res.sort_values().plot.barh()
        plt.title(f"{n}-grams {label_type} labels ({most_common} most common)")
        plt.xlabel("Count")
        plt.ylabel(f"{n}-grams")
    fig.tight_layout()
    plt.savefig(f"plots/{n}_grams_{most_common}.png")
    plt.show()

if __name__ == "__main__":
    df, texts, sources = help.read_seperate_data()
    visualize_ngrams(df, 2, 20)
    visualize_ngrams(df, 3, 20)

    #vocabulary = help.find_most_common(texts, 50, 1)
    #BoW = help.generate_bow(texts, vocabulary)
    #visualize_word_count(df, texts, sources, BoW, vocabulary, exclude_first = 15, n_words = 50)

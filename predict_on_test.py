import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd

from argparse import ArgumentParser
from models import ClassifierRNNs


from dataset import SSTDataset
from load_embed import load_embedding


def load_model(args, emb):
    model = ClassifierRNNs(args, emb)
    model.load_state_dict(torch.load(args.path_model), strict=True)

    return model


def load_data(args,df):
    print("Loading the dataset...")

    vec_model = load_embedding("/cluster/shared/nlpl/data/vectors/latest/" + args.zip_file)
    #vec_model = load_embedding(args.zip_file)
    vec_model.add('<unk>', weights=torch.rand(vec_model.vector_size))
    vec_model.add('<pad>', weights=torch.zeros(vec_model.vector_size))
    pad_idx = vec_model.vocab['<pad>'].index

    test_dataset = SSTDataset(df, vec_model,args.data_type)
    test_iter = DataLoader(test_dataset)

    return test_iter, vec_model

def predict(test_iter, model):
    X,length,y = test_iter
    
    y_pred = model.forward(X,length)
    y_pred = y_pred.max(dim=1)[1]

    return y_pred

def write_to_file(y_pred,df):
    y_pred.numpy()
    y_pred = list(y_pred)
    y_pred = ["negative" if y == 0 else "positive" for y in y_pred]
    df.label = y_pred
    df.to_csv("predictions.tsv.gz")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--test", default="")
    parser.add_argument("--path_model", default="model_1.pt")

    parser.add_argument("--path", default="data/stanford_sentiment_binary.tsv.gz")
    parser.add_argument("--model", default="RNNs") #RNNs or FFNN
    parser.add_argument("--rnn_type", default="RNN") ## Simple RNN, LSTM or GRU

    parser.add_argument("--input_size", action="store", type=int, default=100)
    parser.add_argument("--hidden_dim", action="store", type=int, default=50)
    parser.add_argument("--n_hidden_layers", action="store", type=int, default=1)
    parser.add_argument("--batch_size", action="store", type=int, default=32)
    parser.add_argument("--lr", action="store", type=float, default=1e-3)
    parser.add_argument("--epochs", action="store", type=int, default=60)
    parser.add_argument("--split", action="store", type=float, default=0.9)
    parser.add_argument("--gamma", action="store", type=float, default=0.9)
    parser.add_argument("--zip_file", default="40.zip")
    parser.add_argument("--grid_search", action="store", type=bool, default=False)  

    parser.add_argument("--compose_word_rep", default="mean") #mean or sum for FFNN
    parser.add_argument("--data_type", default="raw") #raw, lemmatized or POS-tagged
    
    parser.add_argument("--bidirectional",type=bool, default=False) #true or false
    parser.add_argument("--dropout",type=float, default=0.0) #float between 0 and 1
    parser.add_argument("--which_state", default="last") #last, max or mean

    
    args = parser.parse_args()

    df = pd.read_csv(args.test, sep='\t', header=0, compression='gzip')

    test_iter, emb = load_data(args,df)

    model = load_model(args, emb)
    y_pred = predict(test_iter, model)

    write_to_file(y_pred,df)







    
















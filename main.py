import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer, AdamW

from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
import pandas as pd
import random, os, logging
import numpy as np

from dataset import SSTDataset
from models import ClassifierMLP,ClassifierRNNs
from load_embed import load_embedding
from evaluate import evaluate, train, pad_batches

def seed_everything(seed_value=5550):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_lemmatised(df, args):
    if args.data_type == "lemmatized":
        just_lemmatized = np.zeros(len(df.lemmatized.values), dtype=object)
        for idx, s in enumerate(df.lemmatized.values):
            just_lemmatized[idx] = " ".join(word.split("_")[0] for word in s.split())

        df["lemmatized_wout_pos"] = just_lemmatized
        
        return df

    return df

def load_data(args):
    print("Loading the dataset...")
    df = pd.read_csv(args.path, sep='\t', header=0, compression='gzip')
    df = make_lemmatised(df,args)
    train_df, val_df = train_test_split(df, train_size=args.split)

    vec_model = load_embedding("/cluster/shared/nlpl/data/vectors/latest/" + args.zip_file)
    vec_model.add('<unk>', weights=torch.rand(vec_model.vector_size))
    vec_model.add('<pad>', weights=torch.zeros(vec_model.vector_size))
    pad_idx = vec_model.vocab['<pad>'].index

    train_dataset = SSTDataset(train_df, vec_model,args.data_type)
    val_dataset = SSTDataset(val_df, vec_model,args.data_type)

    train_iter = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,collate_fn=lambda x: pad_batches(x, pad_index=pad_idx))
    val_iter = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,collate_fn=lambda x: pad_batches(x, pad_index=pad_idx))

    return train_iter, val_iter, vec_model

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
    logger = logging.getLogger(__name__)
    seed_everything(5550)

    parser = ArgumentParser()
    parser.add_argument("--path", default="data/stanford_sentiment_binary.tsv.gz")
    parser.add_argument("--model", default="RNNs") #RNNs or FFNN
    parser.add_argument("--rnn_type", default="GRU") ## Simple RNN, LSTM or GRU
    parser.add_argument("--input_size", action="store", type=int, default=100)
    parser.add_argument("--hidden_dim", action="store", type=int, default=100)
    parser.add_argument("--n_hidden_layers", action="store", type=int, default=1)
    parser.add_argument("--batch_size", action="store", type=int, default=64)
    parser.add_argument("--lr", action="store", type=float, default=1e-2)
    parser.add_argument("--epochs", action="store", type=int, default=50)
    parser.add_argument("--split", action="store", type=float, default=0.9)
    parser.add_argument("--zip_file", default="40.zip")
    parser.add_argument("--grid_search", action="store", type=bool, default=False)  
    parser.add_argument("--compose_word_rep", default="mean") #mean or sum for FFNN
    parser.add_argument("--data_type", default="raw") #raw, lemmatized or POS-tagged    
    parser.add_argument("--bidirectional",type=bool, default=True) #true or false
    parser.add_argument("--dropout",type=float, default=0.1) #float between 0 and 1
    parser.add_argument("--which_state", default="mean") #last, max or mean
    args = parser.parse_args()
    logger.info(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_iter, val_iter, vec_model = load_data(args)
    model = {
        "RNNs": ClassifierRNNs,
        "FFNN": ClassifierMLP
    }[args.model](args,vec_model).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * len(train_iter))


    for epoch in range(args.epochs):
        loss = train(model, criterion, train_iter, optimizer,device)
        scheduler.step()

        train_acc = evaluate(device, model, train_iter)
        test_acc = evaluate(device, model, val_iter)

        logger.info(f"Epoch: {epoch+1:.2f} - Train accuracy: {train_acc} - Test accuracy: {test_acc}")





    


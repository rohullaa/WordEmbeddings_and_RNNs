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

def run_model():
    logger.info(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

        if args.model == "RNNs": 
            if epoch == 10 and test_acc < 0.65:
                break

def grid_search_ffnn():
    for lr in [1e-3]:
        args.lr = lr
        for hidden_dim in [16,32,64]:
            args.hidden_dim = hidden_dim
            for n_hidden_layers in [1,2,4,6]:
                args.n_hidden_layers = n_hidden_layers
                for compose_word_rep in ["mean","sum"]:
                    args.compose_word_rep = compose_word_rep
                    for zip_file in ["82.zip", "40.zip"]:
                        args.zip_file = zip_file

                        if args.zip_file == "40.zip":
                            args.input_size = 100
                        else:
                            args.input_size = 300

                        run_model()


def grid_search_rnn():
    for rnn_type in ["RNN", "LSTM", "GRU"]:
        for which_state in ["last", "max" "mean"]:
            for bidirectional in [True, False]:
                for dropout in [0,0.1,0.2]:
                    for hidden_dim in [25,50,100]:
                        for input_size in [100, 200]:
                            for num_layers in range(1,5):
                                args.rnn_type = rnn_type
                                args.which_state = which_state
                                args.bidirectional = bidirectional
                                args.hidden_dim = hidden_dim
                                args.input_size = input_size
                                args.n_hidden_layers = num_layers
                                args.dropout = dropout
                                run_model()

                                # if best_acc > acc:
                                #     acc = best_acc
                                #     state_dict = {
                                #     'model': model.state_dict(),
                                #     'training_args': args
                                #     }
                                #     torch.save(state_dict, name)

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
    parser.add_argument("--batch_size", action="store", type=int, default=64)
    parser.add_argument("--lr", action="store", type=float, default=1e-2)
    parser.add_argument("--epochs", action="store", type=int, default=50)
    parser.add_argument("--split", action="store", type=float, default=0.9)
    parser.add_argument("--zip_file", default="40.zip")
    parser.add_argument("--data_type", default="raw") #raw, lemmatized or POS-tagged    
    args = parser.parse_args()
    
    train_iter, val_iter, vec_model = load_data(args)

    if args.model == "FFNN": 
        grid_search_ffnn()
    else:
        grid_search_rnn()





    


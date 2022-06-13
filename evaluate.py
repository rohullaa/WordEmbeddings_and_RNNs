import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer, AdamW

from sklearn.model_selection import train_test_split
from sklearn import metrics
from argparse import ArgumentParser
from typing import Optional, List
import pandas as pd
import os
import random
import numpy as np
import torch
import tqdm
import time

from dataset import SSTDataset
from models import ClassifierMLP,ClassifierRNNs
from load_embed import load_embedding


def seed_everything(seed_value=5550):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(model: nn.Module, criterion: nn.Module, train_iter: DataLoader, optimizer: Optimizer):
    model.train()
    for feature_vector,length,label_true in tqdm.tqdm(train_iter):
        optimizer.zero_grad()
        label_true = torch.squeeze(label_true)
        label_pred = model(feature_vector, length)
        loss = criterion(label_pred, label_true)
        loss.backward()
        optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(model: nn.Module, data_iter: DataLoader, labels: Optional[List[str]] = None):
    model.eval()
    labels_true, predictions = [], []
    for feature_vector,length,label_true in tqdm.tqdm(data_iter):
        output = model(feature_vector,length)
        predictions += output.argmax(dim=1).tolist()
        labels_true += label_true.tolist()

    if labels:
        print(metrics.classification_report(labels_true, predictions, target_names=labels))

    return metrics.accuracy_score(labels_true, predictions)

def pad_batches(batch,pad_index):
    longest_sequence = max([input.size(0) for input,_,_ in batch])
    new_input = torch.stack([
        F.pad(input, (0, longest_sequence - input.size(0)),value=pad_index) for input,_,_ in batch
    ])
    new_lengths = torch.stack([length for _,length,_ in batch])
    new_target = torch.stack([target for _,_, target in batch])

    return new_input, new_lengths, new_target


def print_parameters(args):
    print("Parameters:")
    for key,value in vars(args).items():
        print(key,": ", value)

def make_lemmatised(df, args):
    # Remove pos tags
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
    #vec_model = load_embedding(args.zip_file)
    vec_model.add('<unk>', weights=torch.rand(vec_model.vector_size))
    vec_model.add('<pad>', weights=torch.zeros(vec_model.vector_size))
    pad_idx = vec_model.vocab['<pad>'].index

    train_dataset = SSTDataset(train_df, vec_model,args.data_type)
    val_dataset = SSTDataset(val_df, vec_model,args.data_type)

    train_iter = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,collate_fn=lambda x: pad_batches(x, pad_index=pad_idx))
    val_iter = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,collate_fn=lambda x: pad_batches(x, pad_index=pad_idx))

    return train_iter, val_iter, vec_model

def run_model(MODEL, args, train_iter, val_iter, vec_model,filename):
    model = MODEL(args,vec_model)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    # let's not keep the learning rate constant, but decay it by 0.9 after each epoch
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    list_test_acc = []
    list_train_acc = []
    list_epochs = range(args.epochs)

    for epoch in range(args.epochs):
        loss = train(model, criterion, train_iter, optimizer)
        scheduler.step()

        train_acc = evaluate(model, train_iter)
        test_acc = evaluate(model, val_iter)

        #print(f"epoch: {epoch}\tloss: {loss:.3f}\tAccuracy:: {train_acc:.3f}")
        #print(f"Validation Accuracy: {test_acc:.3f}\n")

        result_str = f"Epoch: {epoch+1:.2f} - Train accuracy: {train_acc} - Test accuracy: {test_acc} - Loss: {loss}\n"
        save_results(filename, result_str)

        list_test_acc.append(test_acc)
        list_train_acc.append(train_acc)

    return max(list_test_acc),list_epochs[ (list_test_acc.index(max(list_test_acc)))] +1, model

def save_results(filename,information):
    with open(filename, 'a+') as f:
        f.write(information)

def write_to_file(filename, args):
    with open(filename, 'w') as f:
        f.write("PARAMETERS: \n")
        f.write("\n")
        f.write(f"Input_size: {args.input_size} , hidden_size: {args.hidden_dim}, num_layers: {args.n_hidden_layers} \n")
        f.write(f"Learning rate: {args.lr}, n_epochs: {args.epochs}\n")
        f.write(f"Zip file: {args.zip_file}\n")

        if args.model == "FFNN":
            f.write(f"Compose_word_rep: {args.compose_word_rep}\n")

        if args.model == "RNNs":
            f.write(f"Model: {args.model},  Bidirectional: {args.bidirectional} , Dropout: {args.dropout}\n")
            f.write(f"Type RNN: {args.rnn_type} \n")

        f.write(f"_________________________________________________________________________________________________\n")

def grid_search_ffnn(args, list_zip_files=["82.zip", "40.zip"]):
    for lr in [1e-3]:
        args.lr = lr
        for hidden_dim in [16,32,64]:
            args.hidden_dim = hidden_dim
            for n_hidden_layers in [1,2,4]:
                args.n_hidden_layers = n_hidden_layers
                for compose_word_rep in ["mean","sum"]:
                    args.compose_word_rep = compose_word_rep
                    for zip_file in list_zip_files:
                        args.zip_file = zip_file

                        if args.zip_file == "40.zip":
                            args.input_size = 100
                        else:
                            args.input_size = 300

                        filename = f"results_fnn/{args.data_type}_{zip_file}_{args.hidden_dim}_{args.n_hidden_layers}_{args.lr}_{args.compose_word_rep}.txt"
                        write_to_file(filename,args)

                        start_time = time.time()
                        train_iter, val_iter, vec_model = load_data(args)
                        run_model(ClassifierMLP,args, train_iter, val_iter, vec_model,filename)
                        end_time = time.time()

                        save_results(filename, f"Time: {end_time-start_time}")

def run_one_time(new_zip= "223.zip", new_epochs=20, new_data_type = "lemmatized"):
    args.epochs = new_epochs
    args.zip_file = new_zip
    args.data_type = new_data_type
    train_iter, val_iter, vec_model = load_data(args)
    filename = f"results_fnn/{args.data_type}_{args.zip_file}_{args.hidden_dim}_{args.n_hidden_layers}_{args.lr}_{args.compose_word_rep}.txt"
    best_acc, best_acc_idx = run_model(ClassifierMLP, args, train_iter, val_iter, vec_model,filename)
    print("###########################################################################")
    print(f"Best test accuracy: {best_acc} at epoch: {best_acc_idx}")

def grid_search_rnn(args,types,states,bi_s,hds,ips,nls,zip_file, name):
    args.lr = 1e-2
    args.zip_file = zip_file
    acc = 0

    for rnn_type in types:
        for which_state in states:
            for bidirectional in bi_s:
                for hidden_dim in hds:
                    for input_size in ips:
                        for num_layers in nls:
                            args.rnn_type = rnn_type
                            args.which_state = which_state
                            args.bidirectional = bidirectional
                            args.hidden_dim = hidden_dim
                            args.input_size = input_size
                            args.n_hidden_layers = num_layers

                            filename = f"results_rnn/{args.rnn_type}_{zip_file}_{input_size}_{hidden_dim}_{num_layers}_{bidirectional}_{which_state}.txt"
                            write_to_file(filename,args)

                            start_time = time.time()
                            train_iter, val_iter, vec_model = load_data(args)
                            best_acc,_,model = run_model(ClassifierRNNs,args, train_iter, val_iter, vec_model,filename)
                            end_time = time.time()

                            save_results(filename, f"Time: {end_time-start_time}")

                            #save the model if it has better acc than the previous models
                            if best_acc > acc:
                                acc = best_acc
                                state_dict = {
                                'model': model.state_dict(),
                                'training_args': args
                                }
                                torch.save(state_dict, name)


def get_param_rnn(which_part):
    types = ["RNN", "LSTM", "GRU"]
    states = ["mean", "last", "max"]
    bi_s = [True, False]
    hds = [10,25,50]

    ips = [100]
    nls = [1,3,5]
    
    zip_file = "40.zip" #or 82.zip
    name = "model"

    if which_part == 1:
        return ["RNN"],states,bi_s,hds,ips,[1],zip_file, f"model_{which_part}.pt"
    elif which_part == 2:
        return ["RNN"],states,bi_s,hds,ips,[3],zip_file, f"model_{which_part}.pt"
    elif which_part == 3:
        return ["RNN"],states,bi_s,hds,ips,[5],zip_file, f"model_{which_part}.pt"

    elif which_part == 4:
        return ["LSTM"],states,bi_s,hds,ips,[1],zip_file, f"model_{which_part}.pt"
    elif which_part == 5:
        return ["LSTM"],states,bi_s,hds,ips,[3],zip_file, f"model_{which_part}.pt"
    elif which_part == 6:
        return ["LSTM"],states,bi_s,hds,ips,[5],zip_file, f"model_{which_part}.pt"

    elif which_part == 7:
        return ["GRU"],states,bi_s,hds,ips,[1],zip_file, f"model_{which_part}.pt"
    elif which_part == 8:
        return ["GRU"],states,bi_s,hds,ips,[3],zip_file, f"model_{which_part}.pt"
    elif which_part == 9:
        return ["GRU"],states,bi_s,hds,ips,[5],zip_file, f"model_{which_part}.pt"

    else:
        print("Error.")

    
if __name__ == "__main__":
    # add command line arguments, this is probably the easiest way to store args for downstream
    parser = ArgumentParser()
    parser.add_argument("--path", default="data/stanford_sentiment_binary.tsv.gz")
    parser.add_argument("--model", default="RNNs") #RNNs or FFNN
    parser.add_argument("--rnn_type", default="RNN") ## Simple RNN, LSTM or GRU

    parser.add_argument("--input_size", action="store", type=int, default=100)
    parser.add_argument("--hidden_dim", action="store", type=int, default=50)
    parser.add_argument("--n_hidden_layers", action="store", type=int, default=3)
    parser.add_argument("--batch_size", action="store", type=int, default=32)
    parser.add_argument("--lr", action="store", type=float, default=1e-2)
    parser.add_argument("--epochs", action="store", type=int, default=60)
    parser.add_argument("--split", action="store", type=float, default=0.9)
    parser.add_argument("--gamma", action="store", type=float, default=0.9)
    parser.add_argument("--zip_file", default="40.zip")
    parser.add_argument("--grid_search", action="store", type=bool, default=False)  

    parser.add_argument("--compose_word_rep", default="mean") #mean or sum for FFNN
    parser.add_argument("--data_type", default="raw") #raw, lemmatized or POS-tagged
    
    parser.add_argument("--bidirectional",type=bool, default=False) #true or false
    parser.add_argument("--dropout",type=float, default=0.0) #float between 0 and 1
    parser.add_argument("--which_state", default="max") #last, max or mean

    args = parser.parse_args()

    # set RNG seed for reproducibility
    seed_everything(5550)


    if args.grid_search == True and args.model == "FFNN":
        grid_search_ffnn(args)

    elif args.grid_search == True and args.model == "RNNs":

        types,states,bi_s,hds,ips,nls,zip_file,name = get_param_rnn(8)

        grid_search_rnn(args,types,states,bi_s,hds,ips,nls,zip_file,name)

    else:

        #run_one_time()
        filename = f"{args.rnn_type}_{args.zip_file}_{args.input_size}_{args.hidden_dim}_{args.n_hidden_layers}_{args.bidirectional}_{args.which_state}_{args.dropout}.txt"
        write_to_file(filename,args)

        start_time = time.time()
        train_iter, val_iter, vec_model = load_data(args)
        best_acc,_,model = run_model(ClassifierRNNs,args, train_iter, val_iter, vec_model,filename)
        end_time = time.time()

        save_results(filename, f"Time: {end_time-start_time}")

        #save the model if it has better acc than the previous models
        if best_acc > 0.72:
            acc = best_acc
            state_dict = {
            'model': model.state_dict(),
            'training_args': args
        }
        torch.save(state_dict, "test_model_best.pt")

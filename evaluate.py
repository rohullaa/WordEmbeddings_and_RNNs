import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from sklearn import metrics
from typing import Optional, List
import torch
import tqdm


def train(model: nn.Module, criterion: nn.Module, train_iter: DataLoader, optimizer: Optimizer, device:torch.device):
    model.train()
    for feature_vector,length,label_true in tqdm.tqdm(train_iter):
        optimizer.zero_grad()
        label_true = torch.squeeze(label_true).to(device)
        label_pred = model(feature_vector, length, device)
        loss = criterion(label_pred, label_true)
        loss.backward()
        optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(model: nn.Module, data_iter: DataLoader, labels: Optional[List[str]] = None):
    model.eval()
    labels_true, predictions = [], []
    for feature_vector,length,label_true in tqdm.tqdm(data_iter):
        output = model(feature_vector,length,device)
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


# def write_to_file(filename, args):
#     with open(filename, 'w') as f:
#         f.write("PARAMETERS: \n")
#         f.write("\n")
#         f.write(f"Input_size: {args.input_size} , hidden_size: {args.hidden_dim}, num_layers: {args.n_hidden_layers} \n")
#         f.write(f"Learning rate: {args.lr}, n_epochs: {args.epochs}\n")
#         f.write(f"Zip file: {args.zip_file}\n")

#         if args.model == "FFNN":
#             f.write(f"Compose_word_rep: {args.compose_word_rep}\n")

#         if args.model == "RNNs":
#             f.write(f"Model: {args.model},  Bidirectional: {args.bidirectional} , Dropout: {args.dropout}\n")
#             f.write(f"Type RNN: {args.rnn_type} \n")

#         f.write(f"_________________________________________________________________________________________________\n")

# def grid_search_ffnn(args, list_zip_files=["82.zip", "40.zip"]):
#     for lr in [1e-3]:
#         args.lr = lr
#         for hidden_dim in [16,32,64]:
#             args.hidden_dim = hidden_dim
#             for n_hidden_layers in [1,2,4]:
#                 args.n_hidden_layers = n_hidden_layers
#                 for compose_word_rep in ["mean","sum"]:
#                     args.compose_word_rep = compose_word_rep
#                     for zip_file in list_zip_files:
#                         args.zip_file = zip_file

#                         if args.zip_file == "40.zip":
#                             args.input_size = 100
#                         else:
#                             args.input_size = 300

#                         filename = f"results_fnn/{args.data_type}_{zip_file}_{args.hidden_dim}_{args.n_hidden_layers}_{args.lr}_{args.compose_word_rep}.txt"
#                         write_to_file(filename,args)

#                         start_time = time.time()
#                         train_iter, val_iter, vec_model = load_data(args)
#                         run_model(ClassifierMLP,args, train_iter, val_iter, vec_model,filename)
#                         end_time = time.time()

#                         save_results(filename, f"Time: {end_time-start_time}")

# def grid_search_rnn(args,types,states,bi_s,hds,ips,nls,zip_file, name):
#     args.lr = 1e-2
#     args.zip_file = zip_file
#     acc = 0

#     for rnn_type in types:
#         for which_state in states:
#             for bidirectional in bi_s:
#                 for hidden_dim in hds:
#                     for input_size in ips:
#                         for num_layers in nls:
#                             args.rnn_type = rnn_type
#                             args.which_state = which_state
#                             args.bidirectional = bidirectional
#                             args.hidden_dim = hidden_dim
#                             args.input_size = input_size
#                             args.n_hidden_layers = num_layers

#                             filename = f"results_rnn/{args.rnn_type}_{zip_file}_{input_size}_{hidden_dim}_{num_layers}_{bidirectional}_{which_state}.txt"
#                             write_to_file(filename,args)

#                             start_time = time.time()
#                             train_iter, val_iter, vec_model = load_data(args)
#                             best_acc,_,model = run_model(ClassifierRNNs,args, train_iter, val_iter, vec_model,filename)
#                             end_time = time.time()

#                             save_results(filename, f"Time: {end_time-start_time}")

#                             #save the model if it has better acc than the previous models
#                             if best_acc > acc:
#                                 acc = best_acc
#                                 state_dict = {
#                                 'model': model.state_dict(),
#                                 'training_args': args
#                                 }
#                                 torch.save(state_dict, name)


# def get_param_rnn(which_part):
#     types = ["RNN", "LSTM", "GRU"]
#     states = ["mean", "last", "max"]
#     bi_s = [True, False]
#     hds = [10,25,50]

#     ips = [100]
#     nls = [1,3,5]
    
#     zip_file = "40.zip" #or 82.zip
#     name = "model"

#     if which_part == 1:
#         return ["RNN"],states,bi_s,hds,ips,[1],zip_file, f"model_{which_part}.pt"
#     elif which_part == 2:
#         return ["RNN"],states,bi_s,hds,ips,[3],zip_file, f"model_{which_part}.pt"
#     elif which_part == 3:
#         return ["RNN"],states,bi_s,hds,ips,[5],zip_file, f"model_{which_part}.pt"

#     elif which_part == 4:
#         return ["LSTM"],states,bi_s,hds,ips,[1],zip_file, f"model_{which_part}.pt"
#     elif which_part == 5:
#         return ["LSTM"],states,bi_s,hds,ips,[3],zip_file, f"model_{which_part}.pt"
#     elif which_part == 6:
#         return ["LSTM"],states,bi_s,hds,ips,[5],zip_file, f"model_{which_part}.pt"

#     elif which_part == 7:
#         return ["GRU"],states,bi_s,hds,ips,[1],zip_file, f"model_{which_part}.pt"
#     elif which_part == 8:
#         return ["GRU"],states,bi_s,hds,ips,[3],zip_file, f"model_{which_part}.pt"
#     elif which_part == 9:
#         return ["GRU"],states,bi_s,hds,ips,[5],zip_file, f"model_{which_part}.pt"

#     else:
#         print("Error.")

    
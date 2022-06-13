import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassifierMLP(nn.Module):
    def __init__(self, args, emb):
        super().__init__()
        self.compose_word_rep = args.compose_word_rep

        self.embedder = nn.Embedding(len(emb.vectors), args.input_size)
        self.input_layer = nn.Linear(args.input_size, args.hidden_dim)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(args.hidden_dim, args.hidden_dim)
            for _ in range(args.n_hidden_layers)
        ])
        self.output_layer = nn.Linear(args.hidden_dim, 2)

    def forward(self, x, length, device):
        x = x.to(device)
        x = self.embedder(x).to(device)
        if self.compose_word_rep == "mean":
            x = torch.mean(x, dim=1).relu()
        elif self.compose_word_rep == "sum":
            x = torch.sum(x, dim=1).relu()

        x = self.input_layer(x).relu()
        for layer in self.hidden_layers:
            x = x + layer(x).relu()

        x = self.output_layer(x)
        return x

class ClassifierRNNs(nn.Module):
    def __init__(self,args, emb):
        super().__init__()
        self.n_hidden_layers = args.n_hidden_layers
        self.hidden_dim = args.hidden_dim
        self.rnn_type = args.rnn_type
        self.bidirectional = args.bidirectional
        self.which_state = args.which_state

        self.embedder = nn.Embedding(len(emb.vectors), args.input_size)

        if self.rnn_type == "RNN":
            self._rnn = nn.RNN(input_size=args.input_size, hidden_size=args.hidden_dim, 
                    num_layers=args.n_hidden_layers,bidirectional = self.bidirectional,
                    dropout = args.dropout, batch_first=True)

        elif self.rnn_type == "GRU":
            self._rnn = nn.GRU(input_size=args.input_size, hidden_size=args.hidden_dim, 
                                num_layers=args.n_hidden_layers,bidirectional = self.bidirectional,
                                dropout = args.dropout, batch_first=True)

        elif self.rnn_type == "LSTM":  
            self._rnn = nn.LSTM(input_size=args.input_size, hidden_size=args.hidden_dim, 
                    num_layers=args.n_hidden_layers,bidirectional = self.bidirectional,
                    dropout = args.dropout, batch_first=True)
        else:
            pass
        
        #the dimensions of the last layer differs dependent of the self.bidirectional
        if self.bidirectional:
            self.bn2 = nn.BatchNorm1d(args.hidden_dim*2)
            self.fc = nn.Linear(args.hidden_dim*2, 2)
        else:    
            
            self.bn2 = nn.BatchNorm1d(args.hidden_dim)
            self.fc = nn.Linear(args.hidden_dim, 2)


    def forward(self, x, length, device):
        x = x.to(device)
        x = self.embedder(x).to(device)

        packed = nn.utils.rnn.pack_padded_sequence(x, length, batch_first=True,enforce_sorted=False)
        #hidden = torch.zeros(self.n_hidden_layers,x.size(0),self.hidden_dim)

        out_packed, _ = self._rnn(packed,None)
        output_rnn, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)

        row_indices = torch.arange(0, x.size(0)).long()
        col_indices = length - 1
        if self.which_state == "last":
            last_tensor=output_rnn[row_indices, col_indices, :]

        elif self.which_state == "max":
            last_tensor = output_rnn[row_indices, :, :]
            last_tensor,_ = torch.max(last_tensor, dim=1)
        elif self.which_state == "mean":
            last_tensor = output_rnn[row_indices, :, :]
            last_tensor = torch.mean(last_tensor, dim=1)

        fc_input = self.bn2(last_tensor)
        out = self.fc(fc_input)

        return out
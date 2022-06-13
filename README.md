# Word Embeddings and Reccurent neural networks (RNNs)

This repo contains implementation of pretrained word embeddings with feed-forward neural networks (FFNN) and RNNs. 

## Motivation 
How can we teach machines to understand text data? We know that machines excel at dealing with and processing numerical data, but when we feed them raw text material, they become into sputtering instruments. The goal is to produce a representation of words that encapsulates their meanings, semantic links, and the various situations in which they are employed. The numerical representation of a text is what word embeddings are.

## Run the code
```
python3 main.py --path [path to data]
                --model [RNNs or FFNN]
                --rnn_type [RNN, GRU or LSTM]
                --input_size INPUT_SIZE
                --hidden_dim HIDDEN_DIM
                --n_hidden_layers N_HIDDEN_LAYERS
                --batch_size BATCH_SIZE
                --lr LR
                --epochs EPOCHS
                --split SPLIT
                --zip_file [zip file for the word embeddings]
                --data_type [raw, lemmatized or POS-tagged]
                --which_state [last, max or mean for RNN model]
                --bidirectional BIDIRECTIONAL
                --dropout DROPOUT
                --compose_word_rep [mean or sum for FFNN model]
```
Fine-tune the hyperparameters of the models:

```
python3 tune_model.py --path PATH
                    --model MODEL
                    --input_size INPUT_SIZE
                    --batch_size BATCH_SIZE
                    --lr LR
                    --epochs EPOCHS
                    --split SPLIT
                    --zip_file ZIP_FILE
                    --data_type DATA_TYPE
```
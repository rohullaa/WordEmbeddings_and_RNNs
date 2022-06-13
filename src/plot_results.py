import matplotlib.pyplot as plt
import numpy as np

def plot_result(model_type, input_type, zip_file, hidden_size, num_layers, lr, flattening_sentence):
    """
    Inputs:
        model_type (str): ffnn or rnn
        input_type (str): pos_tagged or raw
        zip_file (str): 40.zip, 82.zip or 223.zip
        hidden_size (int): 16, 32, 64
        num_layers (int): 1, 2, 4, 6
        lr (int): 0.01, 0.001, 0.0001
        flattening_sentence (str): mean or sum
    Outpus:
        None
    """
    file_name = f"results_{model_type}/{input_type}_{zip_file}_{hidden_size}_{num_layers}_{lr}_{flattening_sentence}.txt"
    train_acc = []
    test_acc = []
    loss = []

    #from IPython import embed; embed()
    with open(file_name, "r") as file:
        line_count = 0
        lines = file.readlines()
        nr_lines = len(lines)

        # Read until we get to the epochs
        for line in lines:
            if "Epoch" not in line:
                line_count += 1
            else:
                break

        for i in range(line_count, nr_lines - 1):
            words = lines[i].split()
            train_acc.append(float(words[5]))
            test_acc.append(float(words[9]))
            loss.append(float(words[12]))

    print(f"Max accuracy: {np.max(test_acc)}")
    plt.plot(np.arange(nr_lines-(line_count+1)), test_acc, label="Accuracy")
    plt.plot(np.arange(nr_lines-(line_count+1)), loss, label="Loss")
    plt.title(f"Accuracy and Loss through epochs, max acc = {np.max(test_acc):.3f}")
    plt.xlabel("Epochs")
    plt.ylabel("Values")
    plt.tight_layout()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_result("fnn", "raw", "40.zip", 16, 2, 0.001, "mean")

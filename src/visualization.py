import matplotlib.pyplot as plt
import numpy as np

def plot_images(x_data, y_data, rows: int, cols: int, predicted_labels=None):
    fig = plt.figure(figsize=(2 * cols, 2 * rows))
    axes = fig.subplots(rows, cols)
    for i, ax in enumerate(axes.flat):
        ax.imshow(x_data[i].squeeze(), cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        if predicted_labels is None:
            ax.set_title('True: %d' % y_data[i])
        else:
            ax.set_title('True: {0}, Pred: {1}'.format(y_data[i], predicted_labels[i]))

def accuracy_plot(train_acc, valid_acc):
    """
    Function to plot the change of accuracy during training.
    """
    # accuracy plots
    plt.figure(figsize=(8, 5))
    plt.plot(
        train_acc, color='red', linestyle='-',
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-',
        label='validation accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

def loss_plot(train_loss, valid_loss):
    """
    Function to plot the change of loss during training.
    """
    # loss plots
    plt.figure(figsize=(8, 5))
    plt.plot(
        train_loss, color='red', linestyle='-',
        label='train loss'
    )
    if valid_loss is not None:
        plt.plot(
            valid_loss, color='blue', linestyle='-',
            label='validation loss'
        )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()


def plot_dropout_change(probs_history):
    probability_changes = np.array(probs_history).T

    colors = ['b', 'g', 'r', 'c']

    plt.figure(figsize=(8, 5))
    for i, layer_prob in enumerate(probability_changes):
        plt.plot(
            layer_prob, color=colors[i], linestyle='dashed',
            label='Dropout layer ' + str(i + 1)
        )

    plt.xlabel('Epochs')
    plt.ylabel('Probability')
    plt.legend()


def plot_accuracy_budget_curves(active_learning_process):
    colors = ['b', 'g', 'r', 'c', 'k']
    plt.figure(figsize=(8, 5))
    c = 0
    for name, data in active_learning_process.items():
        samples = [int(s) for s in data.keys()]
        accuracies = data.values()
        plt.plot(
            samples, accuracies, color=colors[c], linestyle='-',
            label=name
        )
        c += 1
    plt.xlabel('Number of samples')
    plt.ylabel('Test accuracy')
    plt.legend()
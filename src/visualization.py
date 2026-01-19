import matplotlib.pyplot as plt

def plot_images(data, rows: int, cols: int, predicted_labels=None):
    fig = plt.figure(figsize=(2 * cols, 2 * rows))
    axes = fig.subplots(rows, cols)
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i][0].squeeze(), cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        if predicted_labels is None:
            ax.set_title('True: %d' % data[i][1])
        else:
            ax.set_title('True: {0}, Pred: {1}'.format(data[i][1], predicted_labels[i]))

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
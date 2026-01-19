import copy

import torch
from tqdm.notebook import tqdm

from .model import ConcreteDropout


def step(model, data, loss_fn, optimizer=None, test=True):
    """
    Performs one training or test step (depends on parameters).
    :param model:
    :param data:
    :param loss_fn:
    :param optimizer: Just don't put anything if it's a test step.
    :param test: True if test step, False if train step.
    :return: Accuracy of predictions and average loss during the training or testing cycle.
    """
    # Set model for training or testing
    if test:
        model.eval()
    else:
        model.train()

    loss_sum = 0.0
    correct_predictions = 0
    total_samples = 0

    # Set mode for training or testing
    with torch.inference_mode(test):
        for X, y in data:
            X, y = X.to(model.device), y.to(model.device)

            # Forward pass
            y_logits = model(X)

            performance_loss = loss_fn(y_logits, y)

            if not test:
                # Zero gradients
                optimizer.zero_grad()
                # Backpropagation
                performance_loss.backward()
                # Parameters update
                optimizer.step()

            # Making predictions
            yp = torch.argmax(y_logits, dim=1)

            # Accumulate loss
            loss_sum += performance_loss.item() * y.size(0)
            # Accumulate accuracy
            total_samples += y.size(0)
            correct_predictions += (yp == y).sum().item()

    # Calculate average loss and accuracy for the epoch
    loss_sum = loss_sum / total_samples
    acc = correct_predictions / total_samples
    return acc, loss_sum


def train_model(model, train_loader, validation_loader, max_epochs, loss_fn, optimizer, disable_logs=False):
    """
    Trains model for set amount of epochs.
    :param model:
    :param train_loader:
    :param validation_loader:
    :param max_epochs:
    :param loss_fn:
    :param optimizer:
    :param disable_logs: False to print logs, True if not.
    :return: Highest performing checkpoint, Logs with metrics for every epoch.
    """
    best_acc = 0
    best_checkpoint = {}
    training_history = {
        'train_loss': [], 'valid_loss': [],
        'train_acc': [], 'valid_acc': []
    }

    for epoch in tqdm(range(max_epochs), disable=disable_logs):
        # Train step
        train_acc, train_loss_data = step(model, train_loader, loss_fn, optimizer, test=False)

        # Test step
        validation_acc, valid_loss_data = step(model, validation_loader, loss_fn, test=True)

        if not disable_logs:
            print(
                f'After epoch {epoch}, avg training loss is {train_loss_data:.4f}, avg validation loss is {valid_loss_data:.4f}, acc on train set is {train_acc * 100:.2f}% and acc on validation set is {validation_acc * 100:.2f}%')

        # Saving logs
        training_history['train_loss'].append(train_loss_data)
        training_history['valid_loss'].append(valid_loss_data)
        training_history['train_acc'].append(train_acc)
        training_history['valid_acc'].append(validation_acc)

        # Saving model with the highest validation accuracy
        if validation_acc > best_acc:
            best_acc = validation_acc
            best_checkpoint = {
                'epoch': epoch,
                'state_dict': copy.deepcopy(model.state_dict()),
                'accuracy': validation_acc,
                'loss': valid_loss_data,
            }

    return best_checkpoint, training_history


def test_with_sampling(model, data, loss_fn, num_classes, t=50):
    model.eval()

    loss_sum = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.inference_mode():
        for X, y in data:
            # X, y = X.to(model.device), y.to(model.device)

            probs_acc = torch.zeros((X.shape[0], num_classes))
            loss_acc = 0
            # Forward pass with sampling
            for _ in range(t):
                y_logits = model(X)
                y_probs = torch.softmax(y_logits, dim=1)
                probs_acc += y_probs

                performance_loss = loss_fn(y_logits, y)
                loss_acc += performance_loss.item()

            probs_acc = probs_acc / t
            loss_acc = loss_acc / t

            # Making predictions
            yp = torch.argmax(probs_acc, dim=1)

            # Accumulate loss
            loss_sum += loss_acc * y.size(0)
            # Accumulate accuracy
            total_samples += y.size(0)
            correct_predictions += (yp == y).sum().item()

    # Calculate average loss and accuracy for the epoch
    loss_sum = loss_sum / total_samples
    acc = correct_predictions / total_samples
    return acc, loss_sum

def train_model_with_sampling(model, train_loader, validation_loader, max_epochs, loss_fn, optimizer,
                              num_classes, t=50, disable_logs=False):
    best_acc = 0
    best_checkpoint = {}
    training_history = {
        'train_loss': [], 'valid_loss': [],
        'train_acc': [], 'valid_acc': [],
        'probs': []
    }

    for epoch in tqdm(range(max_epochs), disable=disable_logs):
        # Train step
        train_acc, train_loss_data = step(model, train_loader, loss_fn, optimizer, test=False)

        # Test step
        validation_acc, valid_loss_data = test_with_sampling(model, validation_loader, loss_fn, num_classes, t=t)

        if not disable_logs:
            print(
                f'After epoch {epoch}, avg training loss is {train_loss_data:.4f}, avg validation loss is {valid_loss_data:.4f}, acc on train set is {train_acc * 100:.2f}% and acc on validation set is {validation_acc * 100:.2f}%')

        # Saving logs
        training_history['train_loss'].append(train_loss_data)
        training_history['valid_loss'].append(valid_loss_data)
        training_history['train_acc'].append(train_acc)
        training_history['valid_acc'].append(validation_acc)
        print(model.cd1.p, model.cd2.p, model.cd3.p, model.cd4.p)
        training_history['probs'].append([cd.p.cpu().data.numpy()[0] for cd in model.dropout_layers])

        # Saving model with the highest validation accuracy
        if validation_acc > best_acc:
            best_acc = validation_acc
            best_checkpoint = {
                'epoch': epoch,
                'state_dict': copy.deepcopy(model.state_dict()),
                'accuracy': validation_acc,
                'loss': valid_loss_data,
            }

    return best_checkpoint, training_history
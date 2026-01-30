import math
from typing import Callable, Any

import torch
from torch.utils.data import DataLoader
from scipy.stats import entropy
import numpy as np

from . import data, train


class ActiveLearning(object):
    """
    Implementation of the active learning.
    :param model_class: the model class to use for training.
    :param model_params: the model hyperparameters.
    :param x_train, y_train, x_test, y_test: train and test data.
    :param uncertainty_fn: the uncertainty function used to compute model's uncertainty on samples.
    :param initial_train_size: the initial number of samples in the training set.
    :param sample_batch_size: amount of samples added to the training set on each step.
    :param device: device for training.
    """

    def __init__(self,
                 model_class: "Class of model to use for training",
                 model_params: dict,
                 x_train: torch.Tensor,
                 y_train: torch.Tensor,
                 x_test: torch.Tensor,
                 y_test: torch.Tensor,
                 uncertainty_fn: Callable,
                 initial_train_size: int = 1000,
                 sample_batch_size: int = 1000,
                 device: torch.device | str = 'cpu') -> None:
        self.model_class = model_class
        self.model_params = model_params

        self.sample_batch_size = sample_batch_size
        self.device = device

        test_dataset = data.SVHNDataset(x_test, y_test)
        self.test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

        split = int(0.8 * len(x_train))
        x_train, y_train, x_val, y_val = x_train[:split], y_train[:split], x_train[split:], y_train[split:]

        val_dataset = data.SVHNDataset(x_val, y_val)
        self.validation_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

        self.x_pool, x_train = x_train[initial_train_size:], x_train[:initial_train_size]
        self.y_pool, y_train = y_train[initial_train_size:], y_train[:initial_train_size]

        self.train_dataset = data.SVHNDataset(x_train, y_train)

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.uncertainty_fn = uncertainty_fn

        train_loader = DataLoader(self.train_dataset, batch_size=64, shuffle=False, num_workers=4)

        # Training the initial model
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        cnn_model = self.model_class(**self.model_params).to(self.device)
        optimizer = torch.optim.SGD(cnn_model.parameters(), lr=0.001, momentum=0.9)

        checkpoint, self.history = train.train_model(cnn_model, train_loader, self.validation_loader, 200,
                                                     self.loss_fn, optimizer, disable_logs=True)

        # Calculating test accuracy of the initial model
        self.trained_model = self.model_class(**self.model_params).to(self.device)
        self.trained_model.load_state_dict(checkpoint['state_dict'])

        test_acc, test_loss = train.step(self.trained_model, self.test_loader, self.loss_fn, test=True)

        self.active_learning_process = {
            len(self.train_dataset): test_acc
        }

    def step(self) -> None:
        """
        Performs one step of active learning. \n
        1) Calculate uncertainties for samples in the training pool.
        2) Choose self.sample_batch_size of samples with the highest uncertainty.
        3) Add those samples to training set, remove them from the training pool.
        4) Train a model on updated training set.
        5) Compute test accuracy.
        """
        # Calculating uncertainties
        self.trained_model.eval()

        with torch.inference_mode():
            uncertainty_pool = self.uncertainty_fn(self.trained_model, self.x_pool.to(self.device))

        uncertainty_pool = torch.tensor(uncertainty_pool)

        # Choosing most uncertain
        most_uncertain = torch.topk(uncertainty_pool, self.sample_batch_size, sorted=False)

        # Updating training set
        self.train_dataset.extend(np.take(self.x_pool, most_uncertain[1], axis=0),
                                  np.take(self.y_pool, most_uncertain[1], axis=0))
        self.x_pool = np.delete(self.x_pool, most_uncertain[1], axis=0)
        self.y_pool = np.delete(self.y_pool, most_uncertain[1], axis=0)

        # Training a model
        train_loader = DataLoader(self.train_dataset, batch_size=64, shuffle=True, num_workers=4,
                                  generator=torch.Generator().manual_seed(42))

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        cnn_model = self.model_class(**self.model_params).to(self.device)
        optimizer = torch.optim.SGD(cnn_model.parameters(), lr=0.001, momentum=0.9)

        checkpoint, self.history = train.train_model(cnn_model, train_loader, self.validation_loader, 200,
                                                     self.loss_fn, optimizer, disable_logs=True)

        # Computing test accuracy
        self.trained_model = self.model_class(**self.model_params).to(self.device)
        self.trained_model.load_state_dict(checkpoint['state_dict'])

        test_acc, test_loss = train.step(self.trained_model, self.test_loader, self.loss_fn, test=True)

        self.active_learning_process[len(self.train_dataset)] = test_acc


def montecarlo_entropy(model: "Trained model", x_pool: torch.Tensor,
                       num_classes: int = 10, t: int = 50) -> torch.Tensor:
    """
    Calculate given model's uncertainty for samples in the pool
    using entropy of prediction after multiple forward passes with dropout layers ON.
    :param model: trained model.
    :param x_pool: pool of samples.
    :param num_classes: number of classes in prediction.
    :param t: amount of forward passes to perform for each sample.
    :return: uncertainty scores for every sample in the pool.
    """
    for layer in model.dropout_layers:
        layer.train()

    batch_size = 250
    batched_len = math.ceil(len(x_pool) / batch_size)
    uncertainties = torch.empty((0,))
    for i in range(batched_len):
        X = x_pool[i * batch_size:(i + 1) * batch_size]

        probs_acc = torch.zeros((X.shape[0], num_classes)).to(model.device)

        # Forward passes
        for _ in range(t):
            y_logits = model(X)
            y_probs = torch.softmax(y_logits, dim=1)
            probs_acc += y_probs

        probs_acc = probs_acc / t
        uncertainty = torch.tensor(entropy(probs_acc.to('cpu'), axis=1))

        uncertainties = torch.cat((uncertainties, uncertainty), dim=0)

    return uncertainties


def random_baseline(_: Any, x_pool: torch.Tensor) -> torch.Tensor:
    """
    Uses random permutation to simulate random uncertainty scores for samples in the pool.
    :param _: absolutely anything. Here for compatibility.
    :param x_pool: pool of samples.
    :return: "uncertainty scores" for every sample in the pool.
    """
    return torch.randperm(x_pool.shape[0], generator=torch.Generator().manual_seed(42))


def selection_strategy_performance(active_learning_process: dict) -> float:
    """
    Calculates performance of active learning method based on comparing rate of steps, where performance rises,
    and steps, where performance falls.
    :param active_learning_process: data for the accuracy vs budget curve.
    :return: the performance score.
    """
    accuracy = list(active_learning_process.values())

    paulc = 0
    naulc = 0
    tpr = 0
    tnr = 0
    for i in range(len(accuracy) - 1):
        # Calculate the rate of change of the performance during this step
        rate_change = accuracy[i + 1] - accuracy[i]
        # Calculate area under curve for this step
        area_under_curve = (accuracy[i + 1] + accuracy[i]) * 0.5
        # If rate of change is positive - we add to Total Positive Rate and Total Positive Area
        if rate_change > 0:
            tpr += rate_change
            paulc += area_under_curve
        # If rate of change is negative or zero - we add to Total Negative Rate and Total Negative Area
        else:
            tnr += rate_change
            naulc += area_under_curve
    return (paulc * tpr + naulc * tnr) / (len(accuracy) - 1)


def area_under_budget_curve(active_learning_process: dict) -> float:
    """
    Calculates performance of active learning method based on area under accuracy vs budget curve.
    :param active_learning_process: data for the accuracy vs budget curve.
    :return: the performance score.
    """
    accuracy = list(active_learning_process.values())
    aubc = 0
    for i in range(len(accuracy) - 1):
        # Calculate area under curve for this step
        area_under_curve = (accuracy[i + 1] + accuracy[i]) * 0.5
        aubc += area_under_curve
    return aubc / (len(accuracy) - 1)


def max_accuracy(active_learning_process: dict) -> (float, int):
    """
    Returns the maximum accuracy reached by the active learning method and the budget with which it was done.
    :param active_learning_process: data for the accuracy vs budget curve.
    :return: (max_accuracy, budget)
    """
    budget = list(active_learning_process.keys())
    accuracy = list(active_learning_process.values())

    max_idx = np.argmax(accuracy)
    return accuracy[max_idx], budget[max_idx]

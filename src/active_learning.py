import math

import torch
from torch.utils.data import DataLoader
from scipy.stats import entropy
import numpy as np

from . import data, train


class ActiveLearning(object):
    def __init__(self, model_class, model_params, x_train, y_train, x_test, y_test, uncertainty_fn,
                 initial_train_size=1000, sample_batch_size=1000, device='cpu'):

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

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        cnn_model = self.model_class(**self.model_params).to(self.device)
        optimizer = torch.optim.SGD(cnn_model.parameters(), lr=0.001, momentum=0.9)

        checkpoint, self.history = train.train_model(cnn_model, train_loader, self.validation_loader, 200,
                                                self.loss_fn, optimizer, disable_logs=True)

        self.trained_model = self.model_class(**self.model_params).to(self.device)
        self.trained_model.load_state_dict(checkpoint['state_dict'])

        test_acc, test_loss = train.step(self.trained_model, self.test_loader, self.loss_fn, test=True)

        self.active_learning_process = {
            len(self.train_dataset): test_acc
        }

    def step(self):
        self.trained_model.eval()

        with torch.inference_mode():
            uncertainty_pool = self.uncertainty_fn(self.trained_model, self.x_pool.to(self.device))

        uncertainty_pool = torch.tensor(uncertainty_pool)

        most_uncertain = torch.topk(uncertainty_pool, self.sample_batch_size, sorted=False)

        self.train_dataset.extend(np.take(self.x_pool, most_uncertain[1], axis=0),
                                  np.take(self.y_pool, most_uncertain[1], axis=0))
        self.x_pool = np.delete(self.x_pool, most_uncertain[1], axis=0)
        self.y_pool = np.delete(self.y_pool, most_uncertain[1], axis=0)

        train_loader = DataLoader(self.train_dataset, batch_size=64, shuffle=True, num_workers=4,
                                  generator=torch.Generator().manual_seed(42))

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        cnn_model = self.model_class(**self.model_params).to(self.device)
        optimizer = torch.optim.SGD(cnn_model.parameters(), lr=0.001, momentum=0.9)

        checkpoint, self.history = train.train_model(cnn_model, train_loader, self.validation_loader, 200,
                                                self.loss_fn, optimizer, disable_logs=True)

        self.trained_model = self.model_class(**self.model_params).to(self.device)
        self.trained_model.load_state_dict(checkpoint['state_dict'])

        test_acc, test_loss = train.step(self.trained_model, self.test_loader, self.loss_fn, test=True)

        self.active_learning_process[len(self.train_dataset)] = test_acc


def montecarlo_entropy(model, x_pool, num_classes=10, t=50):
    for layer in model.dropout_layers:
        layer.train()

    batch_size = 250
    batched_len = math.ceil(len(x_pool) / batch_size)
    uncertainties = torch.empty((0,))
    for i in range(batched_len):
        X = x_pool[i * batch_size:(i + 1) * batch_size]

        probs_acc = torch.zeros((X.shape[0], num_classes)).to(model.device)
        # Forward pass with sampling
        for _ in range(t):
            y_logits = model(X)
            y_probs = torch.softmax(y_logits, dim=1)
            probs_acc += y_probs

        probs_acc = probs_acc / t
        uncertainty = torch.tensor(entropy(probs_acc.to('cpu'), axis=1))

        uncertainties = torch.cat((uncertainties, uncertainty), dim=0)

    return uncertainties


def random_baseline(model, x_pool):
    return torch.randperm(x_pool.shape[0])


def selection_strategy_performance(active_learning_process):
    accuracy = list(active_learning_process.values())

    paulc = 0
    naulc = 0
    tpr = 0
    tnr = 0
    for i in range(len(accuracy) - 1):
        rate_change = accuracy[i + 1] - accuracy[i]
        area_under_curve = (accuracy[i + 1] + accuracy[i]) * 0.5
        if rate_change > 0:
            tpr += rate_change
            paulc += area_under_curve
        else:
            tnr += rate_change
            naulc += area_under_curve
    return (paulc * tpr + naulc * tnr) / (len(accuracy) - 1)


def area_under_budget_curve(active_learning_process):
    accuracy = list(active_learning_process.values())
    aubc = 0
    for i in range(len(accuracy) - 1):
        area_under_curve = (accuracy[i + 1] + accuracy[i]) * 0.5
        aubc += area_under_curve
    return aubc / (len(accuracy) - 1)


def max_accuracy(active_learning_process):
    budget = list(active_learning_process.keys())
    accuracy = list(active_learning_process.values())

    max_idx = np.argmax(accuracy)
    return accuracy[max_idx], budget[max_idx]
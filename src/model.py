import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class DropoutCNN(nn.Module):
    def __init__(self, dropout):
        super(DropoutCNN, self).__init__()

        self.dropout = nn.Dropout(dropout)

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=1, padding='same'),
            nn.ELU(),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1, padding='same'),
            nn.ELU(),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding='same'),
            nn.ELU(),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Flatten()
        )
        self.fc1 = nn.Linear(in_features=1024, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=10)

    def forward(self, x):
        x = self.block1(x)
        x = self.dropout(x)

        x = self.block2(x)
        x = self.dropout(x)

        x = self.block3(x)
        x = self.dropout(x)

        x = F.elu(self.fc1(x))
        x = self.dropout(x)

        x = F.elu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

    def predict_proba(self, x):
        x = self.forward(x)
        return F.softmax(x, dim=1).numpy()


class SOLDropoutCNN(DropoutCNN):
    def __init__(self, dropout):
        super(SOLDropoutCNN, self).__init__(dropout)

    def forward(self, x):
        x = self.block1(x)
        x = self.dropout(x)

        x = self.block2(x)
        x = self.dropout(x)

        x = self.block3(x)
        x = self.dropout(x)

        x = F.elu(self.fc1(x))
        x = self.dropout(x)

        x = F.elu(self.fc2(x))
        return self.fc3(x)


class ConcreteDropout(nn.Module):
    """Concrete Dropout.

    Implementation of the Concrete Dropout module as described in the
    'Concrete Dropout' paper: https://arxiv.org/pdf/1705.07832
    """

    def __init__(self, weight_regularizer: float, dropout_regularizer: float,
                 init_min: float = 0.2, init_max: float = 0.3) -> None:

        """Concrete Dropout.

        Parameters
        ----------
        weight_regularizer : float
        dropout_regularizer : float
        init_min : float
        init_max : float
        """

        super().__init__()

        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer

        init_min = np.log(init_min) - np.log(1.0 - init_min)
        init_max = np.log(init_max) - np.log(1.0 - init_max)

        self.p_logit = nn.Parameter(torch.empty(1).uniform_(init_min, init_max))
        self.p = torch.sigmoid(self.p_logit)

        self.regularisation = 0.0

    def forward(self, x, layer):

        """Calculates the forward pass.

        The regularisation term for the layer is calculated and assigned to a
        class attribute - this can later be accessed to evaluate the loss.

        Parameters
        ----------
        x : Tensor
            Input to the Concrete Dropout.
        layer : nn.Module
            Layer for which to calculate the Concrete Dropout.

        Returns
        -------
        Tensor
            Output from the dropout layer.
        """

        output = layer(self._concrete_dropout(x))

        sum_of_squares = 0
        for param in layer.parameters():
            sum_of_squares += torch.sum(torch.pow(param, 2))

        weights_reg = self.weight_regularizer * sum_of_squares / (1.0 - self.p)

        dropout_reg = self.p * torch.log(self.p)
        dropout_reg += (1.0 - self.p) * torch.log(1.0 - self.p)
        dropout_reg *= self.dropout_regularizer * x.shape[1]

        self.regularisation = weights_reg + dropout_reg

        return output

    def _concrete_dropout(self, x):

        """Computes the Concrete Dropout.

        Parameters
        ----------
        x : Tensor
            Input tensor to the Concrete Dropout layer.

        Returns
        -------
        Tensor
            Outputs from Concrete Dropout.
        """

        eps = 1e-7
        tmp = 0.1

        self.p = torch.sigmoid(self.p_logit)

        if len(x.shape) == 2:
            u_noise = torch.rand_like(x)
        elif len(x.shape) == 4:
            u_shape = (x.shape[0], x.shape[1], 1, 1)
            u_noise = torch.rand(u_shape)

        drop_prob = torch.sigmoid((self.p_logit + torch.log(u_noise + eps) - torch.log(1 - u_noise + eps)) / tmp)

        random_tensor = 1 - drop_prob
        retain_prob = 1 - self.p

        x = torch.mul(x, random_tensor) / retain_prob

        return x


class ConcreteDropoutCNN(nn.Module):
    def __init__(self):
        super(ConcreteDropoutCNN, self).__init__()

        w, d = 1e-6, 1e-3

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=1, padding='same'),
            nn.ELU(),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )

        self.dropout_layers = nn.ModuleList()
        self.dropout_layers.append(ConcreteDropout(weight_regularizer=w, dropout_regularizer=d))

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1, padding='same'),
            nn.ELU(),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )

        self.dropout_layers.append(ConcreteDropout(weight_regularizer=w, dropout_regularizer=d))

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding='same'),
            nn.ELU(),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Flatten()
        )

        self.dropout_layers.append(ConcreteDropout(weight_regularizer=w, dropout_regularizer=d))

        self.fc1 = nn.Linear(in_features=1024, out_features=128)

        self.dropout_layers.append(ConcreteDropout(weight_regularizer=w, dropout_regularizer=d))

        self.fc2 = nn.Linear(in_features=128, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=10)

        self.elu = nn.ELU()

    def forward(self, x):
        x = self.block1(x)

        x = self.dropout_layers[0](x, self.block2)

        x = self.dropout_layers[1](x, self.block3)

        x = self.dropout_layers[2](x, nn.Sequential(self.fc1, self.elu))

        x = self.dropout_layers[3](x, nn.Sequential(self.fc2, self.elu))

        return self.fc3(x)

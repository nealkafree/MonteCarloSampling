import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class DropoutCNN(nn.Module):
    """
    Implementation of the convolutional neural network for usage with Monte Carlo Dropout.
    :param dropout: dropout probability for all layers.
    """

    def __init__(self, dropout: float) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    @property
    def device(self) -> torch.device:
        """
        Returns the device on which the model is.
        """
        return next(self.parameters()).device


class SOLDropoutCNN(DropoutCNN):
    """
    Version of DropoutCNN without dropout layer after penultimate layer.
    :param dropout: dropout probability for all layers.
    """

    def __init__(self, dropout) -> None:
        super(SOLDropoutCNN, self).__init__(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    """
    Implementation of the Concrete Dropout module.
    :param weight_regularizer: regularizer for weight regularization parameter.
    :param dropout_regularizer: regularizer for dropout regularization parameter.
    :param init_min: minimal probability value for parameter initialization.
    :param init_max: maximal probability value for parameter initialization.
    """

    def __init__(self, weight_regularizer: float, dropout_regularizer: float,
                 init_min: float = 0.1, init_max: float = 0.1) -> None:
        super().__init__()

        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer

        init_min = np.log(init_min) - np.log(1.0 - init_min)
        init_max = np.log(init_max) - np.log(1.0 - init_max)

        self.p_logit = nn.Parameter(torch.empty(1).uniform_(init_min, init_max))
        self.p = torch.sigmoid(self.p_logit)

        self.regularisation = 0.0

    def forward(self, x: torch.Tensor, layer: nn.Module) -> torch.Tensor:
        """
        Calculates the forward pass through Concrete Dropout layer.
        :param x: input to the Concrete Dropout layer.
        :param layer: layer (or block) after the Concrete Dropout layer, used to calculate regularisation.
        :return: output after the aforementioned layer (or block).
        """
        if self.training:
            # Calculating output
            output = layer(self._concrete_dropout(x))

            # Calculating weight regularization
            sum_of_squares = 0
            for param in layer.parameters():
                sum_of_squares += torch.sum(torch.pow(param, 2))
            weights_reg = self.weight_regularizer * sum_of_squares / (1.0 - self.p)

            # Calculating dropout regularization
            dropout_reg = self.p * torch.log(self.p)
            dropout_reg += (1.0 - self.p) * torch.log(1.0 - self.p)
            dropout_reg *= self.dropout_regularizer * x.shape[1]

            # Saving regularization for this layer
            self.regularisation = weights_reg + dropout_reg
        else:
            output = layer(x)

        return output

    def _concrete_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the Concrete Dropout.
        :param x: input tensor to the Concrete Dropout layer.
        :return: output from Concrete Dropout layer.
        """
        # Constant for numerical stability
        eps = 1e-7
        # Temperature parameter. Effect is untested.
        tmp = 0.1

        self.p = torch.sigmoid(self.p_logit)

        # Implementation for dense layers
        if len(x.shape) == 2:
            u_noise = torch.rand_like(x).to(self.p_logit.device)
        # Implementation for convolutional layers
        elif len(x.shape) == 4:
            u_shape = (x.shape[0], x.shape[1], 1, 1)
            u_noise = torch.rand(u_shape).to(self.p_logit.device)

        # Calculating what is to be dropped
        drop_prob = torch.sigmoid((self.p_logit + torch.log(u_noise + eps) - torch.log(1 - u_noise + eps)) / tmp)

        random_tensor = 1 - drop_prob
        retain_prob = 1 - self.p

        # Calculating outputs
        x = torch.mul(x, random_tensor) / retain_prob

        return x


class ConcreteDropoutCNN(nn.Module):
    """
    Implementation of convolutional neural network which uses Concrete Dropout.
    :param weight_regularization: regularizer for weight regularization parameter.
    :param dropout_regularization: regularizer for dropout regularization parameter.
    """

    def __init__(self, weight_regularization: float, dropout_regularization: float) -> None:
        super(ConcreteDropoutCNN, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=1, padding='same'),
            nn.ELU(),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )

        self.dropout_layers = nn.ModuleList()
        self.dropout_layers.append(ConcreteDropout(weight_regularization, dropout_regularization))

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1, padding='same'),
            nn.ELU(),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )

        self.dropout_layers.append(ConcreteDropout(weight_regularization, dropout_regularization))

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding='same'),
            nn.ELU(),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Flatten()
        )

        self.dropout_layers.append(ConcreteDropout(weight_regularization, dropout_regularization))

        self.fc1 = nn.Linear(in_features=1024, out_features=128)

        self.dropout_layers.append(ConcreteDropout(weight_regularization, dropout_regularization))

        self.fc2 = nn.Linear(in_features=128, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=10)

        self.elu = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)

        x = self.dropout_layers[0](x, self.block2)

        x = self.dropout_layers[1](x, self.block3)

        x = self.dropout_layers[2](x, nn.Sequential(self.fc1, self.elu))

        x = self.dropout_layers[3](x, nn.Sequential(self.fc2, self.elu))

        return self.fc3(x)

    @property
    def device(self) -> torch.device:
        """
        Returns the device on which the model is.
        """
        return next(self.parameters()).device

    def regularisation(self) -> torch.Tensor:
        """
        Calculates Concrete Dropout regularisation by combining regularisation from every layer.
        :return: total Concrete Dropout regularisation.
        """

        total_regularisation = 0
        for layer in self.dropout_layers:
            total_regularisation += layer.regularisation

        return total_regularisation

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """
        Needed for compatibility with scikit-learn API.
        :param x: input data.
        :return: predicted probabilities.
        """
        x = self.forward(x)
        return F.softmax(x, dim=1).to('cpu').numpy()

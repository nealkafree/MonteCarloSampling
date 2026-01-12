from torch import nn


class DropoutCNN(nn.Module):
    def __init__(self, dropout):
        super(DropoutCNN, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

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

        x = self.fc1(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.dropout(x)
        return self.fc3(x)

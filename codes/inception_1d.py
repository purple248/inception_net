import torch
import torch.nn.functional as F
import numpy as np

torch.device('cpu')


class InceptionBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, epsilon=0.001):
        super(InceptionBlock, self).__init__()

        def pad_same(f):
            pad = np.ceil((f - 1) / 2)
            return int(pad)

        # conv 1x1 - is like "average" of previous layers - preserves spatial
        # dimensions, reduces depth
        self.branch1_conv1x1 = torch.nn.Conv1d(in_channels=in_channels, kernel_size=1, out_channels=96, stride=1)
        self.branch1_bn_1 = torch.nn.BatchNorm1d(96, eps=epsilon)
        self.branch1_conv3x3 = torch.nn.Conv1d(in_channels=96, kernel_size=3, out_channels=128, stride=1, padding=pad_same(3))
        self.branch1_bn_2 = torch.nn.BatchNorm1d(128, eps=epsilon)

        self.branch2_conv1x1 = torch.nn.Conv1d(in_channels=in_channels, kernel_size=1, out_channels=16, stride=1)
        self.branch2_bn_1 = torch.nn.BatchNorm1d(16, eps=epsilon)
        self.branch2_conv5x5 = torch.nn.Conv1d(in_channels=16, kernel_size=5, out_channels=32, stride=1, padding=pad_same(5))
        self.branch2_bn_2 = torch.nn.BatchNorm1d(32, eps=epsilon)

        self.branch3_maxpool = torch.nn.MaxPool1d(kernel_size=3, stride=1, padding=pad_same(3))
        self.branch3_conv1x1 = torch.nn.Conv1d(in_channels=in_channels, kernel_size=1, out_channels=32, stride=1)
        self.branch3_bn_1 = torch.nn.BatchNorm1d(32, eps=epsilon)

        self.branch4_conv1x1 = torch.nn.Conv1d(in_channels=in_channels, kernel_size=1, out_channels=64, stride=1)
        self.branch4_bn_1 = torch.nn.BatchNorm1d(64, eps=epsilon)

    def forward(self, x):
        #branch 1:
        #print(x.size())
        branch1 = self.branch1_conv1x1(x)
        branch1 = self.branch1_bn_1(branch1)
        branch1 = F.relu(branch1)

        branch1 = self.branch1_conv3x3(branch1)
        branch1 = self.branch1_bn_2(branch1)
        branch1 = F.relu(branch1)

        # branch 2:
        branch2 = self.branch2_conv1x1(x)
        branch2 = self.branch2_bn_1(branch2)
        branch2 = F.relu(branch2)

        branch2 = self.branch2_conv5x5(branch2)
        branch2 = self.branch2_bn_2(branch2)
        branch2 = F.relu(branch2)

        # branch 3:
        branch3 = self.branch3_maxpool(x)
        branch3 = self.branch3_conv1x1(branch3)
        branch3 = self.branch3_bn_1(branch3)
        branch3 = F.relu(branch3)

        # branch 4:
        branch4 = self.branch4_conv1x1(x)
        branch4 = self.branch4_bn_1(branch4)
        branch4 = F.relu(branch4)

        # out_channels = 256
        return torch.cat([branch1, branch2, branch3, branch4], 1)

# define the model:
class CNN_LSTM_model(torch.nn.Module):

    def __init__(self, n_features, hidden_dim, n_layers, in_channels, cnn_out_channels, n_labels):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.cnn_out_channels1, self.cnn_out_channels2, self.cnn_out_channels3, self.cnn_out_channels4 = cnn_out_channels

        # define net structure:
        self.inception1 = InceptionBlock(n_features, 256, epsilon=0.001)
        self.inception2 = InceptionBlock(256, 256, epsilon=0.001)
        self.inception3 = InceptionBlock(256, 256, epsilon=0.001)
        self.inception4 = InceptionBlock(256, 256, epsilon=0.001)

        self.lstm = torch.nn.LSTM(256, hidden_dim, n_layers, batch_first=True)  # batch_first=True: rnn input shape = (batch_size, seq_len, features), rnn output shape = (batch_size, seq_len, hidden_size)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, n_labels)

    def init_hidden(self, batch_size):
        # generate the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden

    def forward(self, X):
        # X: batch data, tensor type, shape = (batch_size, seq_len, features)
        batch_size = X.size(0)

        # cnn net:
        X = X.transpose(1, 2)
        X = self.inception1(X)
        X = self.inception2(X)
        X = self.inception3(X)
        X = self.inception4(X)

        # fix axis order:
        X = X.transpose(1, 2)

        # lstm step
        hidden_state = self.init_hidden(batch_size)
        cell_state = self.init_hidden(batch_size)
        X, _ = self.lstm(X, (hidden_state, cell_state)) # out shape = (batch_size, seq_len, hidden_size)
        X = X[:, -1, :]

        # fc step
        X = self.bn2(X)
        X = self.fc(X)
        X = torch.sigmoid(X)

        return X.view(-1, )

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from prepare import get_word_to_token, get_word_to_vector

word2token = get_word_to_token()
word2vec = get_word_to_vector(word2token)


class TextCNN(nn.Module):
    def __init__(self, max_length):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(len(word2token), 50)
        self.embedding.weight.requires_grad = True
        self.embedding.weight.data.copy_(torch.from_numpy(word2vec))
        self.maxLength = max_length
        self.name = 'TextCNN'

        """
        There are three parallel convolution layers, one dropout layer and one full-connected layer in total.
        Convolution layers all have 20 kernels with height 3, 5, 7 respectively and width 50 (vector dimension).
        Probability of dropout layer is set to 0.3
        Full-connected layer has 20 * 3 input neurons and 2 output neurons
        """
        self.conv1 = nn.Conv2d(1, 20, (3, 50))
        self.conv2 = nn.Conv2d(1, 20, (5, 50))
        self.conv3 = nn.Conv2d(1, 20, (7, 50))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(60, 2)

    def forward(self, x):
        # embed the sentence token from embedding table, x.size = (1, max_length, 50) (neglecting batch size)
        x = self.embedding(x.to(torch.int64)).unsqueeze(1)

        # convolut by three layers and squeeze, xi.size = (20, max_length - kernel_height[i] + 1)
        # ReLU and max pooling, xi.size = (20)
        x1 = F.max_pool1d(F.relu(self.conv1(x).squeeze(3)), self.maxLength - 2).squeeze(2)
        x2 = F.max_pool1d(F.relu(self.conv2(x).squeeze(3)), self.maxLength - 4).squeeze(2)
        x3 = F.max_pool1d(F.relu(self.conv3(x).squeeze(3)), self.maxLength - 6).squeeze(2)

        # concatenate x1 ~ x3 and dropout, x.size = (60)
        x = self.dropout(torch.cat((x1, x2, x3), 1))

        # pass full-connected layer and get softmax, x.size = (2)
        return F.log_softmax(self.fc(x), dim=1)


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(len(word2token), 50)
        self.embedding.weight.requires_grad = True
        self.embedding.weight.data.copy_(torch.from_numpy(word2vec))
        self.name = 'LSTM'

        """
        The RNN model is implemented using LSTM with multi-head self-attention
        LSTM parameters: input_size = embedding_dim = 50, hidden_size = 100, num_layers = 2, dropout = 0.3
        The weights of attention layer is softmax of output of LSTM layer
        The full-connected layer is used to categorize
        """
        self.lstm = nn.LSTM(input_size=50, hidden_size=100, num_layers=2, dropout=0.5, batch_first=True)
        self.attention = nn.Linear(100, 100)
        self.fc = nn.Linear(100, 2)

    def forward(self, x):
        # embed the sentence token from embedding table and input it into LSTM layer
        output, (h, c) = self.lstm(self.embedding(x.to(torch.int64)))

        # use multi-head self-attention
        attention_weight = F.softmax(self.attention(output), dim=1)
        attention_output = torch.sum(attention_weight * output, dim=1)

        # pass full-connected layer and get the final result
        return self.fc(attention_output)


class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()
        self.embedding = nn.Embedding(len(word2token), 50)
        self.embedding.weight.requires_grad = True
        self.embedding.weight.data.copy_(torch.from_numpy(word2vec))
        self.name = 'GRU'

        """
        RNN_GRU is almost the same as RNN_LSTM externally
        """
        self.gru = nn.GRU(input_size=50, hidden_size=100, num_layers=2, dropout=0.5, batch_first=True)
        self.attention = nn.Linear(100, 100)
        self.fc = nn.Linear(100, 2)

    def forward(self, x):
        # embed the sentence token from embedding table and input it into GRU layer
        output, h = self.gru(self.embedding(x.to(torch.int64)))

        # use multi-head self-attention
        attention_weight = F.softmax(self.attention(output), dim=1)
        attention_output = torch.sum(attention_weight * output, dim=1)

        # pass full-connected layer and get the final result
        return self.fc(attention_output)


class MLP(nn.Module):
    def __init__(self, max_length):
        super(MLP, self).__init__()
        self.embedding = nn.Embedding(len(word2token), 50)
        self.embedding.weight.requires_grad = True
        self.embedding.weight.data.copy_(torch.from_numpy(word2vec))
        self.maxLength = max_length
        self.name = 'MLP'

        """
        There are two linear layers in MLP model
        A dropout layer with probability 0.3 is between the two layers
        """
        self.input = nn.Linear(50, 100)
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(100, 2)

    def forward(self, x):
        x = torch.sigmoid(self.input(self.embedding(x.to(torch.int64)))).permute(0, 2, 1)
        return F.log_softmax(self.output(self.dropout(F.max_pool1d(x, self.maxLength).view(-1, 100))), dim=1)

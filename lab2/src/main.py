import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt
import argparse
import prepare
from models import TextCNN, LSTM, GRU, MLP


def get_data(max_l, batch_s):
    word2token = prepare.get_word_to_token()

    train_texts, train_labels = prepare.get_corpus('../Dataset/train.txt', word2token, max_length=max_l)
    valid_texts, valid_labels = prepare.get_corpus('../Dataset/validation.txt', word2token, max_length=max_l)
    test_texts, test_labels = prepare.get_corpus('../Dataset/test.txt', word2token, max_length=max_l)

    train_dataset = TensorDataset(
        torch.from_numpy(train_texts).type(torch.float),
        torch.from_numpy(train_labels).type(torch.int64)
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_s, shuffle=True, num_workers=2)
    valid_dataset = TensorDataset(
        torch.from_numpy(valid_texts).type(torch.float),
        torch.from_numpy(valid_labels).type(torch.int64)
    )
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_s, shuffle=True, num_workers=2)
    test_dataset = TensorDataset(
        torch.from_numpy(test_texts).type(torch.float),
        torch.from_numpy(test_labels).type(torch.int64)
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_s, shuffle=True, num_workers=2)

    return train_dataloader, valid_dataloader, test_dataloader


def parse_params():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-l', '--initial_learning_rate',
        dest='initial_learning_rate',
        type=float,
        default=1e-3,
        help='initial learning rate, default 1e-3'
    )
    parser.add_argument('-m', '--max_length', dest='max_length', type=int, default=60, help='maximum sentence length')
    parser.add_argument('-e', '--epoch', dest='epoch', type=int, default=10, help='number of epochs to train')
    parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=50, help='batch size')
    parser.add_argument('-n', '--network', dest='network', type=str, default='TextCNN', help='type of model')

    args = parser.parse_args()
    _initial_learning_rate = args.initial_learning_rate
    _max_length = args.max_length
    _epoch = args.epoch
    _batch_size = args.batch_size
    network = args.network

    if network == 'TextCNN':
        _model = TextCNN(_max_length)
    elif network == 'LSTM':
        _model = LSTM()
    elif network == 'GRU':
        _model = GRU()
    elif network == 'MLP':
        _model = MLP(_max_length)
    elif network == 'all':
        _model = nn.ModuleList([MLP(_max_length), TextCNN(_max_length), LSTM(), GRU()])
    else:
        print('Missing or invalid type of neural network')
        exit(1)
    return _initial_learning_rate, _max_length, _epoch, _batch_size, _model


def train(dataLoader, model):
    model.train()
    train_loss, train_acc = [], []
    full_true, full_prediction = [], []
    for index, (x, y) in enumerate(dataLoader):
        x, y = x.to(device), y.to(device)
        prediction = model(x)
        loss = F.cross_entropy(prediction, y)
        correct = torch.eq(prediction.argmax(1), y).to(torch.float32)
        accuracy = correct.sum() / len(correct)
        train_loss.append(loss.item())
        train_acc.append(accuracy.item())
        full_true.extend(y.cpu().numpy().tolist())
        full_prediction.extend(prediction.argmax(1).cpu().numpy().tolist())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss = np.array(train_loss).mean()
    avg_acc = np.array(train_acc).mean()
    f1 = f1_score(np.array(full_true), np.array(full_prediction), average="binary")
    return avg_loss, avg_acc, f1


def evaluate(dataLoader, model):
    model.eval()
    train_loss, train_acc = [], []
    full_true, full_prediction = [], []
    for index, (x, y) in enumerate(dataLoader):
        x, y = x.to(device), y.to(device)
        prediction = model(x)
        loss = F.cross_entropy(prediction, y)
        correct = torch.eq(prediction.argmax(1), y).to(torch.float32)
        accuracy = correct.sum() / len(correct)
        train_loss.append(loss.item())
        train_acc.append(accuracy.item())
        full_true.extend(y.cpu().numpy().tolist())
        full_prediction.extend(prediction.argmax(1).cpu().numpy().tolist())
    avg_loss = np.array(train_loss).mean()
    avg_acc = np.array(train_acc).mean()
    f1 = f1_score(np.array(full_true), np.array(full_prediction), average="binary")
    return avg_loss, avg_acc, f1


if __name__ == '__main__':
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    initial_learning_rate, max_length, epoch, batch_size, model = parse_params()

    if isinstance(model, type(nn.ModuleList())):
        train_dataloader, valid_dataloader, test_dataloader = get_data(max_length, batch_size)
        for m in model:
            optimizer = torch.optim.Adam(m.parameters(), initial_learning_rate)
            train_acc = np.zeros(10)
            train_f1 = np.zeros(10)
            valid_acc = np.zeros(10)
            valid_f1 = np.zeros(10)
            test_acc = np.zeros(10)
            test_f1 = np.zeros(10)
            print(m.name)
            for e in range(epoch):
                avg_loss, avg_acc, f1 = train(train_dataloader, m)
                train_acc[e] = avg_acc
                train_f1[e] = f1
                avg_loss, avg_acc, f1 = evaluate(valid_dataloader, m)
                valid_acc[e] = avg_acc
                valid_f1[e] = f1
                avg_loss, avg_acc, f1 = evaluate(test_dataloader, m)
                test_acc[e] = avg_acc
                test_f1[e] = f1
                print(f"Epoch {e + 1} has finished")

            x = np.linspace(1, 10, 10)

            fig1 = plt.figure(1)
            plt.title('train accuracy')
            plt.xlabel('epoch')
            plt.ylabel('train accuracy')
            plt.plot(x, train_acc, label=m.name)
            plt.legend()

            fig2 = plt.figure(2)
            plt.title('train f1')
            plt.xlabel('epoch')
            plt.ylabel('train f1')
            plt.plot(x, train_f1, label=m.name)
            plt.legend()

            fig3 = plt.figure(3)
            plt.title('validation accuracy')
            plt.xlabel('epoch')
            plt.ylabel('validation accuracy')
            plt.plot(x, valid_acc, label=m.name)
            plt.legend()

            fig4 = plt.figure(4)
            plt.title('validation f1')
            plt.xlabel('epoch')
            plt.ylabel('validation f1')
            plt.plot(x, valid_f1, label=m.name)
            plt.legend()

            fig5 = plt.figure(5)
            plt.title('test accuracy')
            plt.xlabel('epoch')
            plt.ylabel('test accuracy')
            plt.plot(x, test_acc, label=m.name)
            plt.legend()

            fig6 = plt.figure(6)
            plt.title('test f1')
            plt.xlabel('epoch')
            plt.ylabel('test f1')
            plt.plot(x, test_f1, label=m.name)
            plt.legend()

        plt.show()

    else:
        if epoch == 0:
            train_dataloader, valid_dataloader, test_dataloader = get_data(max_length, batch_size)
            optimizer = torch.optim.Adam(model.parameters(), initial_learning_rate)
            acc = np.zeros(30)
            _f1 = np.zeros(30)
            for e in range(1, 31):
                avg_loss, avg_acc, f1 = train(train_dataloader, model)
                avg_loss, avg_acc, f1 = evaluate(test_dataloader, model)
                print(f"Epoch: {e}, test accuracy: {avg_acc}, test f1: {f1}")
                acc[e - 1] = avg_acc
                _f1[e - 1] = f1
            x = np.linspace(1, 30, 30)
            plt.figure()
            plt.title("Accuracy and f1-score for different epochs")
            plt.xlabel('epoch')
            plt.plot(x, acc, label="accuracy")
            plt.plot(x, _f1, label="f1-score")
            plt.legend()
            plt.show()

        elif batch_size == 0:
            x = np.linspace(10, 90, 5)
            plt.figure()
            plt.title("Accuracy for different batch sizes")
            plt.xlabel('epoch')
            plt.ylabel('accuracy')
            for batch_size in x:
                train_dataloader, valid_dataloader, test_dataloader = get_data(max_length, batch_size)
                optimizer = torch.optim.Adam(model.parameters(), initial_learning_rate)
                acc = np.zeros(10)
                for e in range(1, epoch + 1):
                    avg_loss, avg_acc, f1 = train(train_dataloader, model)
                    avg_loss, avg_acc, f1 = evaluate(test_dataloader, model)
                    print(f"test accuracy: {avg_acc}, test f1: {f1}")
                    acc[e - 1] = avg_acc
                plt.plot(x, acc, label=f'{batch_size}')
            plt.legend()
            plt.show()

        elif max_length == 0:
            x = np.linspace(40, 120, 5)
            plt.figure()
            plt.title("Accuracy for different maximum lengths")
            plt.xlabel('epoch')
            plt.ylabel('accuracy')
            for max_length in x:
                max_length = int(max_length)
                train_dataloader, valid_dataloader, test_dataloader = get_data(max_length, batch_size)
                optimizer = torch.optim.Adam(model.parameters(), initial_learning_rate)
                acc = np.zeros(10)
                for e in range(1, epoch + 1):
                    avg_loss, avg_acc, f1 = train(train_dataloader, model)
                    avg_loss, avg_acc, f1 = evaluate(test_dataloader, model)
                    print(f"test accuracy: {avg_acc}, test f1: {f1}")
                    acc[e - 1] = avg_acc
                plt.plot(x, acc, label=f'{max_length}')
            plt.legend()
            plt.show()

        elif initial_learning_rate == 0:
            train_dataloader, valid_dataloader, test_dataloader = get_data(max_length, batch_size)
            x = np.logspace(-1, -5, 5)
            plt.figure()
            plt.title("Accuracy for different initial learning rates")
            plt.xlabel('epoch')
            plt.ylabel('accuracy')
            for initial_learning_rate in x:
                optimizer = torch.optim.Adam(model.parameters(), initial_learning_rate)
                acc = np.zeros(10)
                for e in range(1, epoch + 1):
                    avg_loss, avg_acc, f1 = train(train_dataloader, model)
                    avg_loss, avg_acc, f1 = evaluate(test_dataloader, model)
                    print(f"test accuracy: {avg_acc}, test f1: {f1}")
                    acc[e - 1] = avg_acc
                plt.plot(x, acc, label=f'{initial_learning_rate}')
            plt.legend()
            plt.show()

        else:
            train_dataloader, valid_dataloader, test_dataloader = get_data(max_length, batch_size)
            optimizer = torch.optim.Adam(model.parameters(), initial_learning_rate)
            acc, _f1 = 0, 0
            for e in range(1, epoch + 1):
                avg_loss, avg_acc, f1 = train(train_dataloader, model)
                print(f"Epoch: {e}, train loss: {avg_loss}, train accuracy: {avg_acc}, train f1: {f1}")
                avg_loss, avg_acc, f1 = evaluate(valid_dataloader, model)
                print(f"Epoch: {e}, validation loss: {avg_loss}, validation accuracy: {avg_acc}, validation f1: {f1}")
                avg_loss, avg_acc, f1 = evaluate(test_dataloader, model)
                acc, _f1 = max(acc, avg_acc), max(_f1, f1)
            print(f"Max accuracy: {acc}, Max f1: {_f1}")

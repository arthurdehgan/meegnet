import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as data_split
from mne.io import read_raw_fif
from params import DATA_PATH, SUBJECT_LIST, SAVE_PATH, CHAN_DF, LABELS, MODEL_PATH


N_SUB_PER_BATCH = 4
BATCH_SIZE = 30
SAVE_PATH = SAVE_PATH
N_EPOCHS = 2


def load_subject(sub, data=None, timepoints=2000, ch_type='all'):
    df = pd.read_csv('{}/clean_camcan_participant_data.csv'.format(SAVE_PATH))
    df = df.set_index('Observations')
    gender = (df['gender_code'] - 1)[sub]
    # subject_file = '{}/{}/rest/rest_raw.fif'.format(DATA_PATH, sub)
    subject_file = '{}/{}/rest/rest_raw.fif'.format(DATA_PATH, sub)
    trial = read_raw_fif(subject_file, preload=True).pick_types(meg=True)[:][0]
    if ch_type == 'all':
        mask = [True for _ in range(len(trial))]
        n_channels = 306
    elif ch_type == 'mag':
        mask = CHAN_DF['mag_mask']
        n_channels = 102
    elif ch_type == 'grad':
        mask = CHAN_DF['mag_mask']
        n_channels = 102
    elif ch_type == 'grad':
        mask = CHAN_DF['grad_mask']
        n_channels = 204
    else:
        raise('Error : bad channel type selected')
    trial = trial[mask]
    n_trials = trial.shape[-1] // timepoints
    for i in range(1, n_trials - 1):
        curr = trial[:, i*timepoints:(i+1)*timepoints]
        curr = curr.reshape(1, n_channels, timepoints)
        data = curr if data is None else np.concatenate((data, curr))
    labels = [gender] * (n_trials - 2)
    data = data.astype(np.float32, copy=False)
    return data, labels


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, (1, 2))
        self.pool = nn.MaxPool2d((1, 2), (1, 2))
        self.conv2 = nn.Conv2d(4, 8, (1, 2))
        self.conv3 = nn.Conv2d(8, 16, (1, 2))
        self.conv4 = nn.Conv2d(16, 32, (1, 2))
        self.fc = nn.Linear(404736, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = self.fc(x)
        return x


if __name__ == '__main__':
    net = Net()
    net = net.cuda()
    params = list(net.parameters())
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    X_train, X_test = data_split(SUBJECT_LIST,
                                 shuffle=True,
                                 stratify=LABELS,
                                 random_state=0,
                                 test_size=.1)
    X_train = set(X_train)
    X_test = set(X_test)

    done_sub = set()
    for epoch in range(N_EPOCHS):
        for _ in range(len(X_train)//N_SUB_PER_BATCH):
            sub_list = set(np.random.choice(list(X_train - done_sub),
                                            N_SUB_PER_BATCH,
                                            replace=False))
            print(sub_list)
            done_sub = done_sub.union(sub_list)
            running_loss = 0.0
            train_labels = []
            train_data = None
            for sub in sub_list:
                if train_data is not None:
                    train_data, temp_labels = load_subject(sub,
                                                           train_data,
                                                           ch_type='mag')
                    train_labels += temp_labels
                else:
                    train_data, train_labels = load_subject(sub,
                                                            ch_type='mag')
            train_labels = np.array(train_labels)
            perm = np.random.permutation(range(len(train_data)))
            train_labels = train_labels[perm]
            train_data = train_data[perm]
            for i in range(0, train_data.shape[0], BATCH_SIZE):
                data = train_data[i:i+BATCH_SIZE]
                labels = train_labels[i:i+BATCH_SIZE]
                n, m, t = data.shape
                inputs = data.reshape(n, m, t, 1)
                inputs = np.transpose(inputs, (0, 3, 1, 2))
                inputs = torch.from_numpy(inputs)
                inputs, labels = Variable(inputs), Variable(torch.from_numpy(labels))
                inputs, labels = inputs.cuda(), labels.cuda()
                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.data[0]
                if i % 100 == 0:
                    # print('loss: {:.3f}'.format(running_loss/10))
                    print('[%d, %5d] loss: %.3f' %
                         (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

    done_sub = set()
    correct = 0
    total = 0
    for _ in range(len(X_train)//N_SUB_PER_BATCH):
        sub_list = set(np.random.choice(list(X_test - done_sub),
                                        N_SUB_PER_BATCH,
                                        replace=False))
        print(sub_list)
        done_sub = done_sub.union(sub_list)
        eval_data = None
        eval_labels = []
        for sub in sub_list:
            if eval_data is not None:
                eval_data, temp_labels = load_subject(sub, eval_data)
                eval_labels += temp_labels
            else:
                eval_data, eval_labels = load_subject(sub)

        for i in range(0, eval_data.shape[0], BATCH_SIZE):
            labels = np.array(eval_labels[i:i+BATCH_SIZE])
            data = eval_data[i:i+BATCH_SIZE]
            n, m, t = data.shape
            inputs = data.reshape(n, m, t, 1)
            inputs = np.transpose(inputs, (0, 3, 1, 2))
            inputs = torch.from_numpy(inputs)
            inputs = Variable(inputs).cuda()
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            c = (predicted == labels).squeeze()
            total += labels.shape[0]
            correct += (predicted == labels).sum()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

from __future__ import print_function

import sys
sys.path.append("../") 

import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt
import torch

from utils import *
from agent.bc_agent import BCAgent
from tensorboard_evaluation import Evaluation

if torch.cuda.device_count() > 0:
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

device = torch.device(DEVICE)
print(device)

def read_data(datasets_dir="./data", frac = 0.1):
    """
    This method reads the states and actions recorded in drive_manually.py 
    and splits it into training/ validation set.
    """
    print("... read data")
    data_files = [
        # 'data.pkl_o.gzip'
        'data.pkl_o1.gzip',
        'data.pkl_cp.gzip'
    ]

    state = []
    action = []
    for fl in data_files:
        data_file = os.path.join(datasets_dir, fl)
        f = gzip.open(data_file, 'rb')
        data = pickle.load(f)
        state += data['state']
        action += data['action']

    # get images as features and actions as targets
    X = np.array(state).astype('float32')
    y = np.array(action).astype('float32')

    # split data into training and validation set
    n_samples = X.shape[0]
    X_train, y_train = X[:int((1-frac) * n_samples)], y[:int((1-frac) * n_samples)]
    X_valid, y_valid = X[int((1-frac) * n_samples):], y[int((1-frac) * n_samples):]
    print(X_train.shape)
    print(X_valid.shape)
    return X_train, y_train, X_valid, y_valid


def preprocessing(X_train, y_train, X_valid, y_valid, history_length=1):
    # TODO: preprocess your data here.
    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)
    # 2. you can train your model with discrete actions (as you get them from read_data) by discretizing the action space 
    #    using action_to_id() from utils.py.
    y_train = np.array([action_to_id(y_train[i]) for i in range(y_train.shape[0])])
    y_valid = np.array([action_to_id(y_valid[i]) for i in range(y_valid.shape[0])])
    X_train = rgb2gray(X_train)
    X_valid = rgb2gray(X_valid)
    # History:
    # At first you should only use the current image as input to your network to learn the next action. Then the input states
    # have shape (96, 96, 1). Later, add a history of the last N images to your state so that a state has shape (96, 96, N).
    
    return X_train, y_train, X_valid, y_valid



def slide(data, ids, width):
    return np.array([data[i - width:i] for i in ids])

def sample_minibatch(X, y, batch_size, hl, probs):
    elems = X.shape[0]
    rnd = np.random.choice(range(hl, elems), size=batch_size, replace=True, p=probs)
    # rnd = np.random.randint(low=hl, high=elems, size=batch_size)
    o_x = slide(X, rnd, hl)
    o_y = slide(y, rnd, hl)
    return o_x, o_y

def compute_accuracy(y_out, y_gt):
    true_preds = torch.where(y_out == y_gt, torch.tensor(1).to(device), torch.tensor(0).to(device)).sum()
    all_preds = y_gt.numel()
    return (true_preds.float() / all_preds).item()


def train_model(X_train, y_train, X_valid, y_valid, n_minibatches, batch_size, lr, hl = 1, model_dir="./models", tensorboard_dir="./tensorboard"):
    
    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    train_p, t_weights = get_weights(y_train, hl)
    valid_p, _ = get_weights(y_valid, hl)
    t_probs = train_p / train_p.sum()
    v_probs = valid_p / valid_p.sum()
 
    print("... train model")

    # TODO: specify your agent with the neural network in agents/bc_agent.py 
    agent = BCAgent(device, lr=lr, history_length=hl)
    tensorboard_eval = Evaluation(tensorboard_dir, "imitation_learning", stats=['loss', 'train_accuracy', 'validation_accuracy'])

    # TODO: implement the training
    # 
    # 1. write a method sample_minibatch and perform an update step
    # 2. compute training/ validation accuracy and loss for the batch and visualize them with tensorboard. You can watch the progress of
    #    your training *during* the training in your web browser
    # 
    # training loop
    # for i in range(n_minibatches):
    #     ...
    #     for i % 10 == 0:
    #         # compute training/ validation accuracy and write it to tensorboard
    #         tensorboard_eval.write_episode_data(...)
    print("Starting loop.")
    lbs_sum = {
        'str': 0,
        'r': 0,
        'l': 0,
        'a': 0,
        'b': 0
    }
    for i in range(n_minibatches):
        x, y = sample_minibatch(X_train, y_train, batch_size, hl, t_probs)
        x = torch.tensor(x).to(device)
        y = torch.tensor(y).to(device)
        loss = agent.update(x, y)

        if i % 10 == 0:
            # compute training/ validation accuracy and write it to tensorboard
            print("round " + str(i))
            lbs = count_labls(y[:, hl - 1].cpu())
            lbs_sum['str'] += lbs['str']
            lbs_sum['l'] += lbs['l']
            lbs_sum['r'] += lbs['r']
            lbs_sum['a'] += lbs['a']
            lbs_sum['b'] += lbs['b']
            print(lbs)
            print(lbs_sum)
            outs = agent.predict(x)
            outs = outs.argmax(dim=2)
            train_acc = compute_accuracy(outs, y)

            val_acc_cum = 0
            for ii in range(10):
                inp, lba = sample_minibatch(X_valid, y_valid, batch_size, hl, v_probs)
                inp = torch.tensor(inp).to(device)
                lba = torch.tensor(lba).to(device)
                val_outs = agent.predict(inp)
                val_outs = val_outs.argmax(dim=2)
                val_acc = compute_accuracy(val_outs, lba)
                val_acc_cum += val_acc

            val_acc_cum = val_acc_cum / 10

            eval_dict = {
                "loss": loss.item(),
                "train_accuracy": train_acc,
                "validation_accuracy": val_acc_cum
            }

            print(eval_dict)
            tensorboard_eval.write_episode_data(i, eval_dict)
      
    # TODO: save your agent
    model_dir = agent.save(os.path.join(model_dir, "agent.pt"))
    print("Model saved in file: %s" % model_dir)



def count_labls(ys):
    stra = np.where(ys == 0, 1, 0)
    le = np.where(ys == 1, 1, 0)
    rt = np.where(ys == 2, 1, 0)
    acc = np.where(ys == 3, 1, 0)
    br = np.where(ys == 4, 1, 0)
    ws = stra.sum()
    wl = le.sum()
    wr = rt.sum()
    wa = acc.sum()
    wb = br.sum()
    di = {
        'str': ws,
        'r': wr,
        'l': wl,
        'a': wa,
        'b': wb
    }
    return di

def get_weights(data, hl):
    data = data[hl:]
    cnt = data.shape[0]
    stra = np.where(data == 0, 1, 0)
    le = np.where(data == 1, 1, 0)
    rt = np.where(data == 2, 1, 0)
    acc = np.where(data == 3, 1, 0)
    br = np.where(data == 4, 1, 0)
    ws = stra.sum()
    wl = le.sum()
    wr = rt.sum()
    wa = acc.sum()
    wb = br.sum()
    print('Straight ' + str(ws))
    print('Left ' + str(wl))
    print('Right ' + str(wr))
    print('Accelerate ' + str(wa))
    print('Break ' + str(wb))
    weights = 1 / (np.array([ws, wl, wr, wa, wb]) / cnt)
    print('Weights ' + str(weights))

    re = np.where(stra == 1, weights[0], 0)
    re += np.where(le == 1, weights[1], 0)
    re += np.where(rt == 1, weights[2], 0)
    re += np.where(acc == 1, weights[3], 0)
    re += np.where(br == 1, weights[4], 0)
    return re, weights


if __name__ == "__main__":

    # read data    
    X_train, y_train, X_valid, y_valid = read_data("./data")

    hl = 10
    lr = 1e-4
    batch_size = 32

    # X_train = X_train[:100]
    # X_valid = X_valid[:100]
    # y_train = y_train[:100]
    # y_valid = y_valid[:100]

    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(X_train, y_train, X_valid, y_valid, history_length=hl)

    minibatches = 50000

    # train model (you can change the parameters!)
    train_model(X_train, y_train, X_valid, y_valid, hl=hl, n_minibatches=minibatches, batch_size=batch_size, lr=lr)
 

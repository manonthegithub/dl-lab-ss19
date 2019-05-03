import torch
from model.data import get_data_loader
from model.model import ResNetModel
import matplotlib.pyplot as plt
from torch.nn import functional as F
import json
import os

SNAPSHOT_FILENAME = 'trained_net.model'
VAL_MPJPE_STATE_FN = 'validation_mpjpe_state'
TRAIN_MPJPE_STATE_FN = 'train_mpjpe_state'

RW = os.R_OK + os.W_OK

if torch.cuda.device_count() > 0:
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

device = torch.device(DEVICE)


def create_model(file):
    if os.access(file, RW):
        model = ResNetModel(pretrained=True)
        model.load_state_dict(torch.load(file, map_location=DEVICE))
        print("Snapshot from " + file + " was successfully loaded.")
    else:
        print("There is no access to the snapshot file: " + file)
        print("Initializing new model.")
        model = ResNetModel(pretrained=False)
    model.to(device)
    return model


def process_epoch(model, optimizer, loss_fn, loader, eps, conf, is_train):
    if is_train:
        print("Training")
        model.train()
    else:
        print("Validating")
        model.eval()
        torch.no_grad()
    mpjpe = 0.0
    mpjpe_mean = 0.0
    batches = 0
    for imgs, kps, vs in loader:
        imgs = imgs.to(device)
        kps = kps.to(device)
        vs = vs.to(device).float()

        optimizer.zero_grad()
        pred = model.forward(imgs, '')
        loss = loss_fn(pred, kps).view(conf.batch_size, -1, 2)
        loss = loss * vs.unsqueeze(2)
        lv = loss.sum()

        if is_train:
            lv.backward()
            optimizer.step()

        loss = (loss.sum(dim=2).sqrt().sum(dim=1) / vs.sum(dim=1)).mean()

        loss_d = loss.item()
        mpjpe_mean += loss_d
        mpjpe += loss_d
        batches += 1
        if (batches - 1) % conf.log_every_batches == (conf.log_every_batches - 1):
            mn = mpjpe_mean / conf.log_every_batches
            print('[%d, %5d, %5d] l2 loss: %.3f' % (eps + 1, batches, batches * conf.batch_size, lv.item()))
            print('mpjpe batch mean: %.3f, epoch mean: %.3f' % (mn, mpjpe / batches))
            mpjpe_mean = 0.0

    return mpjpe / batches


def load_mpjpe_state():
    print("Trying to load MPJPE state.")

    if os.access(VAL_MPJPE_STATE_FN, RW) and os.access(TRAIN_MPJPE_STATE_FN, RW):
        vals = torch.load(VAL_MPJPE_STATE_FN)
        train = torch.load(TRAIN_MPJPE_STATE_FN)
        print("Successfully loaded state.")
        return vals, train
    else:
        print("State was not loaded: either the files do not exist or you do not have access.")
        return [],[]


def train(model, optimizer, loss_fn, train_loader, val_loader, conf):
    val_mpjpe, train_mpjpe = load_mpjpe_state()

    for eps in range(conf.epochs):
        train_err = process_epoch(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            loader=train_loader,
            eps=eps,
            conf=conf,
            is_train=True)
        val_err = process_epoch(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            loader=val_loader,
            eps=eps,
            conf=conf,
            is_train=False)
        train_mpjpe.append(train_err)
        val_mpjpe.append(val_err)

        if (conf.take_snapshots_every_epochs is not None) and (
                eps % conf.take_snapshots_every_epochs == (conf.take_snapshots_every_epochs - 1)):
            print("Taking snapshot and saving to: " + conf.path_to_snapshot)
            torch.save(model.state_dict(), conf.path_to_snapshot)
            torch.save(train_mpjpe, TRAIN_MPJPE_STATE_FN)
            torch.save(val_mpjpe, VAL_MPJPE_STATE_FN)
            plot(train_mpjpe, val_mpjpe)
            print("Snapshot was successfully saved to: " + conf.path_to_snapshot)

    print("Finished training.")
    return train_mpjpe, val_mpjpe


def plot(train, val):
    fn = 'task1.png'
    plt.plot(train)
    plt.plot(val)
    plt.legend(['train MPJPE', 'validation MPJPE'], loc='best')
    plt.title('Task1. MPJPE for train and validation sets over epochs.')
    plt.xlabel('epochs')
    plt.ylabel('MPJPE')
    print('Saving error graph in ' + fn)
    plt.savefig(fn, format='png')


def load_conf(conf):
    cfn = 'app.conf'
    if os.access(cfn, os.R_OK):
        conf_j = json.load(open(cfn, 'r'))
        if 'path_to_snapshot' in conf_j:
            conf.path_to_snapshot = conf_j['path_to_snapshot'] + SNAPSHOT_FILENAME
        if 'take_snapshots_every_epochs' in conf_j:
            conf.take_snapshots_every_epochs = conf_j['take_snapshots_every_epochs']
        if 'log_every_batches' in conf_j:
            conf.log_every_batches = conf_j['log_every_batches']
        if 'epochs' in conf_j:
            conf.epochs = conf_j['epochs']
        if 'batch_size' in conf_j:
            conf.batch_size = conf_j['batch_size']
        if 'learning_rate' in conf_j:
            conf.learning_rate = conf_j['learning_rate']
        print('Successfully loaded configuration.')
    else:
        print('Can not read config file ' + cfn)
        print('Using default configuration')
    attrs = vars(conf)
    print(', \n'.join("%s: %s" % item for item in attrs.items()))


class Conf:
    def __init__(self):
        self.take_snapshots_every_epochs = 1
        self.log_every_batches = 40
        self.epochs = 1
        self.batch_size = 25
        self.learning_rate = 1e-4
        self.path_to_snapshot = './' + SNAPSHOT_FILENAME


if __name__ == '__main__':

    conf = Conf()
    load_conf(conf)
    print('Using:' + DEVICE)
    print('Trying to load model snapshot from: ' + conf.path_to_snapshot)
    model = create_model(conf.path_to_snapshot)

    train_loader = get_data_loader(batch_size=conf.batch_size, is_train=True, single_sample=False)
    val_loader = get_data_loader(batch_size=conf.batch_size, is_train=False, single_sample=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate)
    loss_fn = torch.nn.MSELoss(reduction='none')

    train_mpjpe, val_mpjpe = train(model=model,
                                    optimizer=optimizer,
                                    loss_fn=loss_fn,
                                    train_loader=train_loader,
                                    val_loader=val_loader,
                                    conf=conf)

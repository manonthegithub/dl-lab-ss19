import torch
from model.data_seg import get_data_loader
from model.model import SingleUpsampling
from model.model import TripleUpsampling
from model.model import TripleUpsamplingSkip
import matplotlib.pyplot as plt
from torch.nn import functional as F
import json
import os
import sys

SNAPSHOT_FILENAME = 'trained_net_t3.model'
VAL_IOU_STATE_FN = 'validation_iou_state_t3'
TRAIN_IOU_STATE_FN = 'train_iou_state_t3'

RW = os.R_OK + os.W_OK

if torch.cuda.device_count() > 0:
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

device = torch.device(DEVICE)


def create_model(file, type):
    if type == 'single':
        model = SingleUpsampling()
    elif type == 'triple':
        model = TripleUpsampling()
    elif type == 'skip':
        model = TripleUpsamplingSkip()
    else:
        print('Wrong net type: ' + type)
        print('Can be: single, triple, or skip')
        model = None

    if os.access(file, RW):
        model.load_state_dict(torch.load(file, map_location=DEVICE))
        print("Snapshot from " + file + " was successfully loaded.")
    else:
        print("There is no access to the snapshot file: " + file)
        print("Initializing new model.")
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
    iou_cumul = 0.0
    for idx, (imgs, msk) in enumerate(loader):
        imgs = imgs.to(device)
        msk = msk.to(device)

        optimizer.zero_grad()
        pred = model.forward(imgs, '')
        loss = loss_fn(pred, msk)

        if is_train:
            loss.backward()
            optimizer.step()
        pred = pred.round().int()
        msk = msk.int()
        iou = (msk & pred).sum().float() / (msk | pred).sum().float()
        iou_cumul += iou.item()
        if idx % conf.log_every_batches == (conf.log_every_batches - 1):
            print('[%d, %5d, %5d] l2 loss: %.3f' % (eps + 1, idx + 1, (idx + 1) * conf.batch_size, loss.item()))
            print('iou mean over batches: %.3f' % (iou_cumul / (idx + 1)))
    return iou_cumul / len(loader)




def load_iou_state(type):
    print("Trying to load IOU state.")

    if os.access(type + VAL_IOU_STATE_FN, RW) and os.access(type + TRAIN_IOU_STATE_FN, RW):
        vals = torch.load(type + VAL_IOU_STATE_FN)
        train = torch.load(type + TRAIN_IOU_STATE_FN)
        print("Successfully loaded state.")
        return vals, train
    else:
        print("State was not loaded: either the files do not exist or you do not have access.")
        return [],[]


def train(type, model, optimizer, loss_fn, train_loader, val_loader, conf):
    val_iou, train_iou = load_iou_state(type)

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
        train_iou.append(train_err)
        val_iou.append(val_err)

        if (conf.take_snapshots_every_epochs is not None) and (
                eps % conf.take_snapshots_every_epochs == (conf.take_snapshots_every_epochs - 1)):
            print("Taking snapshot and saving to: " + conf.path_to_snapshot)
            torch.save(model.state_dict(), conf.path_to_snapshot)
            torch.save(train_iou, TRAIN_IOU_STATE_FN)
            torch.save(val_iou, VAL_IOU_STATE_FN)
            plot(type, train_iou, val_iou)
            fn = 'task3_' + type + '.png'
            print('Saving loss graph in ' + fn)
            plt.savefig(fn, format='png')
            plt.close()
            print("Snapshot was successfully saved to: " + conf.path_to_snapshot)

    print("Finished training.")
    return train_iou, val_iou


def plot(ln, train, val):
    plt.plot(train, label='train IOU ' + ln)
    plt.plot(val, label='validation IOU ' + ln)
    plt.ylim(bottom=0)
    plt.legend(loc='best')
    plt.title('Task3. IOU for train and validation sets over epochs.')
    plt.xlabel('epochs')
    plt.ylabel('IOU')
    plt.grid(True)


def load_conf(conf, tp):
    cfn = 'app_t3.conf'
    if os.access(cfn, os.R_OK):
        conf_j = json.load(open(cfn, 'r'))
        if 'path_to_snapshot' in conf_j:
            conf.path_to_snapshot = conf_j['path_to_snapshot'] + SNAPSHOT_FILENAME + tp
        else:
            conf.path_to_snapshot = conf.path_to_snapshot + tp
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
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == 'plot':
            tps = [('single', 'single upsampling'), ('triple', 'transposed convolutions'), ('skip', 'transposed convolutions + skip')]

            for t, ttl in tps:
                val, train = load_iou_state(t)
                if len(val) > 0 and len(train) > 0:
                    plot(ttl, train, val)
            fn = 'task3_comparison_all.png'
            print('Saving loss graph in ' + fn)
            plt.savefig(fn, format='png')
            plt.close()
        else:
            tp = sys.argv[1]
            conf = Conf()
            load_conf(conf, tp)
            print('Using:' + DEVICE)
            print('Trying to load model snapshot from: ' + conf.path_to_snapshot)
            model = create_model(conf.path_to_snapshot, tp)

            train_loader = get_data_loader(batch_size=conf.batch_size, is_train=True)
            val_loader = get_data_loader(batch_size=conf.batch_size, is_train=False)
            optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate)
            loss_fn = torch.nn.BCELoss()
            train_iou, val_iou = train(model=model,
                                            optimizer=optimizer,
                                            loss_fn=loss_fn,
                                            train_loader=train_loader,
                                            val_loader=val_loader,
                                            conf=conf,
                                            type=tp)
    else:
        print("Submitted no arguments")

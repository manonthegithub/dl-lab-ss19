import torch
from model.data import get_data_loader
from model.model import ResNetModel
import matplotlib.pyplot as plt
from torch.nn import functional as F
import os

SNAPSHOT_FILENAME = 'trained_net.model'
PATH_TO_SNAPSHOT = './' + SNAPSHOT_FILENAME

if torch.cuda.device_count() > 0:
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

device = torch.device(DEVICE)

take_snapshots_every_epochs = 1
log_every_batches = 100

epochs = 1
batch_size = 25
learning_rate = 1e-4


def create_model(file):
    if os.access(file, os.X_OK):
        model = ResNetModel(pretrained=True)
        model.load_state_dict(torch.load(file, map_location=DEVICE))
        print("Snapshot from " + file + " was successfully loaded.")
    else:
        print("There is no access to the snapshot file: " + file)
        print("Initializing new model.")
        model = ResNetModel(pretrained=False)
    model.to(device)
    return model


def process_epoch(model, optimizer, loss_fn, loader, eps, log_every_batches, is_train):
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
    for batch_id, (imgs, kps, vs) in enumerate(loader):
        imgs = imgs.to(device)
        kps = kps.to(device)
        vs = vs.to(device).float().sum(dim=1)

        optimizer.zero_grad()
        pred = model.forward(imgs, '')
        loss = loss_fn(pred, kps)

        if is_train:
            if loss.dim() > 1:
                loss_n = loss.sum(dim=1).mean()
            else:
                loss_n = loss.mean()
            loss_n.backward()
            optimizer.step()

        ### data specific code
        loss = (loss.view(batch_size, 2, -1).sum(dim=1).sqrt().sum(dim=1) / vs).mean()
        mpjpe_mean += loss.item()

        if batch_id % log_every_batches == (log_every_batches - 1):
            mn = mpjpe_mean / log_every_batches
            mpjpe += mpjpe_mean
            print('[%d, %5d] loss: %.3f' % (eps + 1, batch_id + 1, mn))
            mpjpe_mean = 0.0
        batches += 1
        ### data specific code

    return mpjpe / batches

VAL_MPJPE_STATE_FN = 'validation_mpjpe_state'
TRAIN_MPJPE_STATE_FN = 'train_mpjpe_state'

def load_mpjpe_state():
    print("Trying to load MPJPE state.")
    if os.access(VAL_MPJPE_STATE_FN, os.X_OK) and os.access(TRAIN_MPJPE_STATE_FN, os.X_OK):
        vals = torch.load(VAL_MPJPE_STATE_FN)
        train = torch.load(TRAIN_MPJPE_STATE_FN)
        print("Successfully loaded state.")
        return vals, train
    else:
        print("State was not loaded: either the files do not exist or you do not have access.")
        return [],[]


def train(model, optimizer, loss_fn, train_loader, val_loader, epochs, snapshot_path, take_snapshots_every_epochs,
          log_every_batches):
    train_mpjpe ,val_mpjpe = load_mpjpe_state()

    for eps in range(epochs):
        train_err = process_epoch(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            loader=train_loader,
            eps=eps,
            log_every_batches=log_every_batches,
            is_train=True)
        val_err = process_epoch(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            loader=val_loader,
            eps=eps,
            log_every_batches=log_every_batches,
            is_train=False)
        train_mpjpe.append(train_err)
        val_mpjpe.append(val_err)

        if (take_snapshots_every_epochs is not None) and (
                eps % take_snapshots_every_epochs == (take_snapshots_every_epochs - 1)):
            print("Taking snapshot and saving to: " + snapshot_path)
            torch.save(model.state_dict(), snapshot_path)
            torch.save(train_mpjpe, TRAIN_MPJPE_STATE_FN)
            torch.save(val_mpjpe, VAL_MPJPE_STATE_FN)
            print("Snapshot was successfully saved to: " + snapshot_path)

    print("Finished training.")
    return train_mpjpe, val_mpjpe


def plot(train, val):
    plt.plot(train)
    plt.plot(val)
    plt.legend(['train MPJPE', 'validation MPJPE'], loc='best')
    plt.title('Task1. MPJPE for train and validation sets over epochs.')
    plt.xlabel('epochs')
    plt.ylabel('MPJPE')
    plt.savefig('task1.png', format='png')


if __name__ == '__main__':
    print('Using:' + DEVICE)
    print('Trying to load model snapshot from: ' + PATH_TO_SNAPSHOT)
    model = create_model(PATH_TO_SNAPSHOT)
    train_loader = get_data_loader(batch_size=batch_size, is_train=True, single_sample=False)
    val_loader = get_data_loader(batch_size=batch_size, is_train=False, single_sample=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss(reduction='none')

    (train_mpjpe, val_mpjpe) = train(model=model,
                                    optimizer=optimizer,
                                    loss_fn=loss_fn,
                                    train_loader=train_loader,
                                    val_loader=val_loader,
                                    epochs=epochs,
                                    take_snapshots_every_epochs=take_snapshots_every_epochs,
                                    snapshot_path=PATH_TO_SNAPSHOT,
                                    log_every_batches=log_every_batches)

    plot(train_mpjpe, val_mpjpe)

import torch
from model.data import get_data_loader
from model.model import ResNetModel
from torch.nn import functional as F
import os

SNAPSHOT_FILENAME = 'trained_net.model'
PATH_TO_SNAPSHOT = './' + SNAPSHOT_FILENAME

if torch.cuda.device_count() > 1:
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

device = torch.device(DEVICE)

take_snapshots_every_batches = 20
log_every_batches = 2

epochs = 10
batch_size = 50
learning_rate = 1e-4


def create_model(file):
    if os.access(file, os.R_OK):
        model = ResNetModel(pretrained=True)
        model.load_state_dict(torch.load(file, map_location=DEVICE))
        print("Snapshot from " + file + " was successfully loaded.")
    else:
        print("There is no access to the snapshot file: " + file)
        print("Initializing new model.")
        model = ResNetModel(pretrained=False)
    return model


def process_epoch(model, optimizer, loss_fn, loader, eps, snapshot_path, log_every_batches, is_train,
                  take_snapshots_every_batches=None):
    if is_train:
        print("Training")
        model.train()
    else:
        print("Validating")
        model.eval()
    mpjpe = 0.0
    mpjpe_mean = 0.0
    for batch_id, (imgs, kps, vs) in enumerate(loader):
        imgs = imgs.to(device)
        kps = kps.to(device)
        vs = vs.to(device)

        optimizer.zero_grad()
        pred = model.forward(imgs, '')
        loss = loss_fn(pred, kps)

        if is_train:
            if loss.dim() > 1:
                loss.sum(dim=1).mean().backward()
            else:
                loss.mean().backward()
            optimizer.step()

        ### data specific code
        mpjpe_batch_mean = (loss.view(batch_size, 2, -1).sum(dim=1).sqrt().sum(dim=1) / vs.float().sum(dim=1)).mean()

        mpjpe_mean += mpjpe_batch_mean.item()

        if batch_id % log_every_batches == (log_every_batches - 1):
            mn = mpjpe_mean / log_every_batches
            mpjpe += mpjpe_mean
            print('[%d, %5d] loss: %.3f' % (eps + 1, batch_id + 1, mn))
            mpjpe_mean = 0.0
        ### data specific code
        if is_train:
            if (take_snapshots_every_batches is not None) and (
                    eps % take_snapshots_every_batches == (take_snapshots_every_batches - 1)):
                print("Taking snapshot and saving to: " + snapshot_path)
                torch.save(model.state_dict(), snapshot_path)
                print("Snapshot was successfully saved to: " + snapshot_path)
    return mpjpe / loader.size()


def train(model, optimizer, loss_fn, train_loader, val_loader, epochs, snapshot_path, take_snapshots_every_batches,
          log_every_batches):
    test_mpjpe = []
    val_mpjpe = []
    for eps in range(epochs):
        test_err = process_epoch(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            loader=train_loader,
            eps=eps,
            snapshot_path=snapshot_path,
            take_snapshots_every_batches=take_snapshots_every_batches,
            log_every_batches=log_every_batches,
            is_train=True)
        val_err = process_epoch(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            loader=val_loader,
            eps=eps,
            snapshot_path=snapshot_path,
            log_every_batches=log_every_batches,
            is_train=False)
        test_mpjpe.append(test_err)
        val_mpjpe.append(val_err)
    print("Finished training.")
    return test_mpjpe, val_mpjpe


if __name__ == '__main__':
    print('Using:' + DEVICE)
    print('Trying to load model snapshot from: ' + PATH_TO_SNAPSHOT)
    model = create_model(PATH_TO_SNAPSHOT)
    train_loader = get_data_loader(batch_size=batch_size, is_train=True, single_sample=False)
    val_loader = get_data_loader(batch_size=batch_size, is_train=False, single_sample=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss(reduction='none')

    (test_mpjpe, val_mpjpe) = train(model=model,
                                    optimizer=optimizer,
                                    loss_fn=loss_fn,
                                    train_loader=train_loader,
                                    val_loader=val_loader,
                                    epochs=epochs,
                                    snapshot_path=PATH_TO_SNAPSHOT,
                                    take_snapshots_every_batches=take_snapshots_every_batches,
                                    log_every_batches=log_every_batches)
    print(test_mpjpe, val_mpjpe)

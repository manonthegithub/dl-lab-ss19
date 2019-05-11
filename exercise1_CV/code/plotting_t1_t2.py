from model.model import SoftResNetModel
from model.model import ResNetModel
from model.data import get_data_loader as get_data_loader_hpe
from utils.plot_util import plot_keypoints
import numpy as np
import matplotlib.pyplot as plt
import os
import torch

if torch.cuda.device_count() > 0:
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

device = torch.device(DEVICE)


def plot(li):
    plt.figure()
    for val, tr, val_l, tr_l in li:
        plt.plot(tr, label=tr_l)
        plt.plot(val, label=val_l)
        plt.ylim(bottom=0)
    fn = 'task1+2.png'
    plt.title('Task 1+2. MPJPE for train and validation sets over epochs comparison.')
    plt.xlabel('epochs')
    plt.ylabel('MPJPE')
    plt.grid(True)
    plt.legend(loc='best')
    print('Saving loss graph in ' + fn)
    # plt.savefig(fn, format='png')


def load_mpjpe_state(val, train):
    print("Trying to load MPJPE state " + val + ' ' + train)
    vals = torch.load(val)
    train = torch.load(train)
    print("Successfully loaded state")
    return vals, train


def create_model_soft_res_net():
    file = 'trained_net_t2.model'
    print("Loading SoftResNetModel")
    model = SoftResNetModel(pretrained=False)
    if os.access(file, os.R_OK):
        model.load_state_dict(torch.load(file, map_location=DEVICE))
        print("Snapshot from " + file + " was successfully loaded.")
    else:
        print("ERROR: Could not load the model.")
        print("Using newly initialized model.")
    model.to(device)
    return model


def create_model_regression():
    file = 'trained_net.model'
    print("Loading ResNetModel")
    model = ResNetModel(pretrained=False)
    if os.access(file, os.R_OK):
        model.load_state_dict(torch.load(file, map_location=DEVICE))
        print("Snapshot from " + file + " was successfully loaded.")
    else:
        print("ERROR: Could not load the model.")
        print("Using newly initialized model.")
    model.to(device)
    return model


if __name__ == '__main__':

    reader = get_data_loader_hpe()

    resnet_t2 = create_model_soft_res_net()
    regression = create_model_regression()

    v1, t1 = load_mpjpe_state('validation_mpjpe_state', 'train_mpjpe_state')
    v1p, t1p = load_mpjpe_state('validation_mpjpe_state_pretrained', 'train_mpjpe_state_pretrained')
    v2, t2 = load_mpjpe_state('validation_mpjpe_state_t2', 'train_mpjpe_state_t2')

    li = [
        (v1, t1, 'ResNetModel. validation', 'ResNetModel. train'),
        (v1p, t1p, 'ResNetModel + pretrained ImageNet. validation', 'ResNetModel + pretrained on ImageNet. train'),
        (v2, t2, 'Softargmax. validation', 'Softargmax. train')
    ]

    plot(li)


    for img, keypoints, weights in reader:
        print('img', type(img), img.shape)
        print('keypoints', type(keypoints), keypoints.shape)
        print('weights', type(weights), weights.shape)

        pred_soft, prob_maps = resnet_t2(img, '')
        pred_soft = pred_soft.detach().numpy()
        prob_maps = prob_maps.detach().numpy()

        pred_regr = regression(img, '')
        pred_regr = pred_regr.detach().numpy()


        # turn image tensor into numpy array containing correctly scaled RGB image
        img_rgb = ((np.array(img) + 1.0)*127.5).round().astype(np.uint8).transpose([0, 2, 3, 1])

        # show
        axs = plt.figure().subplots(nrows=1, ncols=3)
        axs[0].imshow(img_rgb[0])
        axs[0].axis('off')
        plot_keypoints(axs[0], keypoints[0], weights[0], draw_limbs=True, draw_kp=True)

        axs[1].imshow(img_rgb[0])
        axs[1].axis('off')
        plot_keypoints(axs[1], pred_soft[0], weights[0], draw_limbs=True, draw_kp=True)

        axs[2].imshow(img_rgb[0])
        axs[2].axis('off')
        plot_keypoints(axs[2], pred_regr[0], weights[0], draw_limbs=True, draw_kp=True)

        axs = plt.figure().subplots(nrows=3, ncols=6)
        idx = 0
        for i in axs:
            for ax in i:
                if(idx < prob_maps.shape[1]):
                    ax.imshow(prob_maps[0][idx]); ax.axis('off')
                ax.axis('off')
                idx += 1

        break
    plt.show()

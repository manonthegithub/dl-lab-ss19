from model.model import SingleUpsampling
from model.model import TripleUpsampling
from model.model import TripleUpsamplingSkip
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
    fn = 'task3.png'
    plt.title('Task 3. IOU for train and validation sets over epochs comparison.')
    plt.xlabel('epochs')
    plt.ylabel('MPJPE')
    plt.grid(True)
    plt.legend(loc='best')
    print('Saving loss graph in ' + fn)
    #plt.savefig(fn, format='png')


def load_iou_state(type):
    vp = 'validation_iou_state_t3'
    tp = 'train_iou_state_t3'
    val = type + vp
    train = type + tp
    print("Trying to load IOU state " + val + ' ' + train)
    vals = torch.load(val)
    train = torch.load(train)
    print("Successfully loaded state")
    return vals, train


def create_model(type):
    pref = 'trained_net_t3.model'
    file = pref + type
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

    if os.access(file, os.R_OK):
        model.load_state_dict(torch.load(file + type, map_location=DEVICE))
        print("Snapshot from " + file + type + " was successfully loaded.")
    else:
        print("There is no access to the snapshot file: " + file + type)
        print("Initializing new model.")
    model.to(device)
    print(model.__class__)
    return model


if __name__ == '__main__':

    reader = get_data_loader_hpe()

    single = create_model('single')
    triple = create_model('triple')
    skip = create_model('skip')

    v_si, t_si = load_iou_state('single')
    v_tr, t_tr = load_iou_state('triple')
    v_sk, t_sk = load_iou_state('skip')

    li = [
        (v_si, v_si, 'Singple upsampling. validation', 'Single upsampling. train'),
        (v_tr, v_tr, '3 layers Transposed Conv. validation', '3 layers Transposed Conv. train'),
        (v_sk, t_sk, '3 layers Transposed Conv + skip. validation', '3 layers Transposed Conv + skip. train')
    ]

    plot(li)


    for img, keypoints, weights in reader:
        print('img', type(img), img.shape)
        print('keypoints', type(keypoints), keypoints.shape)
        print('weights', type(weights), weights.shape)

        pred_si = single(img, '')
        pred_si = pred_si.detach().numpy()

        pred_tr = triple(img, '')
        pred_tr = pred_tr.detach().numpy()

        pred_sk = skip(img, '')
        pred_sk = pred_sk.detach().numpy()


        # turn image tensor into numpy array containing correctly scaled RGB image
        img_rgb = ((np.array(img) + 1.0)*127.5).round().astype(np.uint8).transpose([0, 2, 3, 1])

        # show
        axs = plt.figure().subplots(nrows=1, ncols=4)
        axs[0].imshow(img_rgb[0])
        axs[0].axis('off')
        plot_keypoints(axs[0], keypoints[0], weights[0], draw_limbs=True, draw_kp=True)

        axs[1].imshow(img_rgb[0])
        axs[1].axis('off')
        plot_keypoints(axs[1], pred_si[0], weights[0], draw_limbs=True, draw_kp=True)

        axs[2].imshow(img_rgb[0])
        axs[2].axis('off')
        plot_keypoints(axs[2], pred_tr[0], weights[0], draw_limbs=True, draw_kp=True)

        axs[3].imshow(img_rgb[0])
        axs[3].axis('off')
        plot_keypoints(axs[3], pred_sk[0], weights[0], draw_limbs=True, draw_kp=True)
        break

    plt.show()

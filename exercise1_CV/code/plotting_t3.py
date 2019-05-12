from model.model import SingleUpsampling
from model.model import TripleUpsampling
from model.model import TripleUpsamplingSkip
from model.data_seg import get_data_loader as get_data_loader_seg
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
        plt.plot(tr[:80], label=tr_l)
        plt.plot(val[:80], label=val_l)

    fn = 'task3.png'
    plt.title('Task 3. IOU for train and validation sets over epochs comparison.')
    plt.xlabel('epochs')
    plt.ylabel('IoU')
    plt.ylim(bottom=0)
    plt.grid(True)
    plt.legend(loc='best')
    print('Saving loss graph in ' + fn)
    plt.savefig('t3_compare.png', format='png')


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
        model.load_state_dict(torch.load(file, map_location=DEVICE))
        print("Snapshot from " + file + " was successfully loaded.")
    else:
        print("There is no access to the snapshot file: " + file)
        print("Initializing new model.")
    model.to(device)
    print(model.__class__)
    return model


if __name__ == '__main__':

    reader = get_data_loader_seg()

    single = create_model('single')
    triple = create_model('triple')
    skip = create_model('skip')

    v_si, t_si = load_iou_state('single')
    v_tr, t_tr = load_iou_state('triple')
    v_sk, t_sk = load_iou_state('skip')

    li = [
        (v_si, t_si, 'Singple upsampling. validation', 'Single upsampling. train'),
        (v_tr, t_tr, '3 layers Transposed Conv. validation', '3 layers Transposed Conv. train'),
        (v_sk, t_sk, '3 layers Transposed Conv + skip. validation', '3 layers Transposed Conv + skip. train')
    ]

    plot(li)

    ind = 0
    for img, msk in reader:
        if ind > 8:
            print('img', type(img), img.shape)

            pred_si = single(img, '')
            pred_si = pred_si.detach().numpy()

            pred_tr = triple(img, '')
            pred_tr = pred_tr.detach().numpy()

            pred_sk = skip(img, '')
            pred_sk = pred_sk.detach().numpy()


            # turn image tensor into numpy array containing correctly scaled RGB image
            img_rgb = ((np.array(img) + 1.0)*127.5).round().astype(np.uint8).transpose([0, 2, 3, 1])

            # show
            f = plt.figure(figsize=(12, 7))
            axs = f.subplots(nrows=2, ncols=4)
            axs[0][0].set_title('Ground truth')
            axs[0][0].imshow(img_rgb[0])
            axs[0][0].axis('off')
            axs[1][0].imshow(msk[0])
            axs[1][0].axis('off')

            axs[0][1].set_title('Single upsampling')
            axs[0][1].imshow(img_rgb[0])
            axs[0][1].axis('off')
            axs[1][1].imshow(pred_si[0])
            axs[1][1].axis('off')

            axs[0][2].set_title('3 transposed conv')
            axs[0][2].imshow(img_rgb[0])
            axs[0][2].axis('off')
            axs[1][2].imshow(pred_tr[0])
            axs[1][2].axis('off')

            axs[0][3].set_title('3 transposed conv + skip')
            axs[0][3].imshow(img_rgb[0])
            axs[0][3].axis('off')
            axs[1][3].imshow(pred_sk[0])
            axs[1][3].axis('off')

            plt.tight_layout()
            plt.savefig('t3_qualitative.png', format='png')
            plt.show()
            break
        else:
            ind += 1

import matplotlib.pyplot as plt
import numpy as np
import gzip
import pickle
import os
import tensorflow as tf
import json



def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def extract_data(dict, fn):
    f = tf.train.summary_iterator(fn)
    for v in f:
        for s in v.summary.value:
            if s.tag in dict.keys():
                dict[s.tag][0].append(s.simple_value)
    re = {}
    for _, (r, name) in dict.items():
        re[name] = r
    return re

def imit_dict(hv):
     return {
        'loss_1': ([], 'loss (history ' + str(hv) + ')'),
        'train_accuracy_1': ([], 'train accuracy (history ' + str(hv) + ')'),
        'validation_accuracy_1': ([], 'validation accuracy (history ' + str(hv) + ')')
    }


def merge_dicts(ds):
    re = {}
    for d in ds:
        for l, r in d.items():
            re[l] = r
    return re

def plot(fls, fn, fig_name):
    dict = {
        'episode_reward_1': ([], 'episode_reward')
    }

    dict2 = {
        'episode_reward_1': ([], 'episode_reward')
    }

    plt.figure()
    test = extract_data(dict2, fls[1][0])['episode_reward']
    eval = extract_data(dict, fls[0][0])['episode_reward']

    plt.plot(range(0, len(test), 20), eval, label=fls[0][1])

    plt.plot(moving_average(test, 10), label=fls[1][1])
    plt.legend(loc='best')
    plt.xlabel('episodes')
    plt.ylabel('reward')
    plt.suptitle(fig_name)
    plt.savefig(fn, dpi=100, orientation='portrait', format='png')



if __name__ == "__main__":

    data_files = [
        ('imitation_learning/tensorboard/imitation_learning-20190605-002050_hist10/events.out.tfevents.1559686851.tfpool33', imit_dict(10)),
        ('imitation_learning/tensorboard/imitation_learning-20190605-003926_hist5/events.out.tfevents.1559687967.tfpool33',  imit_dict(5)),
        ('/Users/Kirill/PycharmProjects/dl-lab-ss19/exercise3_R/imitation_learning/tensorboard/imitation_learning-20190605-165202/events.out.tfevents.1559746322.tfpool33', imit_dict(1))
    ]

    all = merge_dicts([extract_data(r, l) for l, r in data_files])

    step = 10

    fg, axs = plt.subplots(len(data_files), len(data_files[0][1].items()))
    inches = 4
    fg.set_size_inches((inches * axs.shape[1], inches * axs.shape[0]))
    axs = axs.reshape(-1)
    for i, (l, r) in enumerate(all.items()):
        axs[i].set_title(l)
        axs[i].set_xlabel('batches')
        if i % len(data_files) == 0:
            axs[i].set_ylim(ymin=0, ymax=2)
        else:
            axs[i].set_ylim(ymin=0)
        av = moving_average(r, 30)
        axs[i].plot(range(0, len(av) * step, step), av, label=l)

    fg.tight_layout()
    # fg.subplots_adjust(top=0.92, bottom=0.15, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
    plt.savefig('imitation.png', dpi=100, orientation='portrait', format='png')

    # fls = [
    #     ('/Users/Kirill/PycharmProjects/dl-lab-ss19/exercise3_R/imitation_learning/results/results_bc_agent-20190605-014432_hist10.json', 10),
    #     ('/Users/Kirill/PycharmProjects/dl-lab-ss19/exercise3_R/imitation_learning/results/results_bc_agent-20190605-015439_hist5.json', 5),
    #     ('/Users/Kirill/PycharmProjects/dl-lab-ss19/exercise3_R/imitation_learning/results/results_bc_agent-20190605-020458_hist1.json', 1)
    # ]
    #
    # plt.figure()
    # for f, hv in fls:
    #     with open(f) as json_file:
    #         data = json.load(json_file)
    #         data['mean']

    fls = [
        ('/Users/Kirill/PycharmProjects/dl-lab-ss19/exercise3_R/reinforcement_learning/tensorboard/eval/cart_pole_eval-20190605-172128/events.out.tfevents.1559748088.fp-10-126-132-106.eduroam-fp.privat', 'eval'),
        ('/Users/Kirill/PycharmProjects/dl-lab-ss19/exercise3_R/reinforcement_learning/tensorboard/train/cart_pole_train-20190605-172128/events.out.tfevents.1559748088.fp-10-126-132-106.eduroam-fp.privat', 'train')
    ]

    plot(fls, 'cartpole.png', 'Cartpole performance')


    fls2 = [
        ('/Users/Kirill/PycharmProjects/dl-lab-ss19/exercise3_R/reinforcement_learning/tensorboard/eval/eval-20190604-221625/events.out.tfevents.1559679398.tfpool57','eval'),
        ('/Users/Kirill/PycharmProjects/dl-lab-ss19/exercise3_R/reinforcement_learning/tensorboard/train/train-20190604-221625/events.out.tfevents.1559679389.tfpool57','train'),

    ]
    plot(fls2, 'carrace.png', 'Carracing performance')
    # plt.subplots(1)
    plt.show()








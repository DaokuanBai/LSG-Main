import os
from collections import OrderedDict
import copy
import numpy as np
import random
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
import options


def set_random_seed(seed=66666):
    """666 is my favorite number."""
    torch.manual_seed(seed + 1)
    torch.cuda.manual_seed(seed + 2)
    # torch.cuda.manual_seed_all(seed + 3) multiple GPUs
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed + 5)
    random.seed(seed + 6)


def sample_by_class(targets, num_class):
    class_idcs = [[] for i in range(num_class)]
    class_idcs_s = [[] for i in range(num_class)]
    for j in range(len(targets)):
        class_idcs[targets[j]].append(j)
    for ii in range(num_class):
        class_idcs_s[ii] = class_idcs[ii]
    return class_idcs, class_idcs_s


def data_noisifying(args, noise_traget, targets, class_idcs, samples):

    num_class = len(class_idcs)

    noise_ratio = args.noise_ratio

    noise_num = []

    for iii in range(num_class):
        noise_num.append(int(len(class_idcs[iii]) * noise_ratio))


    noise_idcs = [[] for i in range(num_class)]

    for j in range(len(noise_idcs)):
        noise_idcs[j] = np.random.choice(int(len(class_idcs[j])), noise_num[j], replace=False)

    n = np.array([])

    for m in range(num_class):
        n = np.concatenate((n, np.array(class_idcs[m])[noise_idcs[m]])).astype(np.int32)

    noise_idcs = n

    for k in range(len(targets)):
        if k in noise_idcs:
            targets[k] = noisify_label(targets[k], args.num_class, args.noise_type)
            # a = list(samples[k])
            # a[1] = targets[k]
            # samples[k] = tuple(a) #When the dataset is Image_net,you are going to need it
            noise_traget[k] = targets[k]


def noisify_label(true_label, num_classes=10, noise_type="symmetric"):
    if noise_type == "symmetric":
        label_lst = list(range(num_classes))
        label_lst.remove(true_label)
        return random.sample(label_lst, k=1)[0]

    elif noise_type == "pairflip":
        return (true_label - 1) % num_classes



def iid_sampling(class_idcs, n_train, num_workers, seed):

    # client_class_idcs = [[] for i in range(len(class_idcs))]
    client_idcs = [[] for i in range(num_workers)]
    for i in range(len(class_idcs)):
        class_idxs = np.array([i for i in range(len(class_idcs[i]))])
        class_random_idcs = np.random.choice(len(class_idcs[i]), len(class_idcs[i]), replace=False)
        class_idcs_permuted = class_idxs[class_random_idcs]
        client_class_idcs = np.array_split(class_idcs_permuted, num_workers)
        for j in range(num_workers):
            client_class = np.array(class_idcs[i])[client_class_idcs[j]]
            client_class = client_class.astype(np.int32)
            client_idcs[j] = np.concatenate((client_idcs[j], client_class)).astype(np.int32)
    return client_idcs

def Clothing_sampling(n_train, num_workers):

    all_idxs=np.array([i for i in range(n_train)]) # initial user and index for whole dataset
    random_idcs = np.random.choice(n_train, n_train, replace=False)# 'replace=False' make sure that there is no repeat
    idcs_permuted = all_idxs[random_idcs]
    client_idcs = np.split(idcs_permuted, num_workers)
    return client_idcs

def n_iid_sampling(class_idcs, n_train, num_workers, seed):
    client_idcs = [np.array([]) for i in range(num_workers)]
    local_datasize = int(n_train / num_workers)
    np.random.seed(seed)
    classes_per_client = 5
    Num_classes = 10
    labels_array = [i for i in range(Num_classes)]
    for i in range(num_workers):
        selected_labels = np.random.choice(labels_array, classes_per_client, replace=False)
        client_distribution = np.array([0. for j in labels_array])
        for each in selected_labels:
            client_distribution[each] = random.uniform(0, 1)
        client_distribution /= client_distribution.sum()
        Bias_array = client_distribution
        Bias_num_array = copy.deepcopy(client_distribution)
        for k in range(len(client_distribution)):
            Bias_num_array[k] = int(local_datasize * Bias_array[k])
            client_idcs[i] = np.concatenate((client_idcs[i], np.random.choice(class_idcs[k], int(Bias_num_array[k]))),
                                            axis=0)
    for i in range(len(client_idcs)):
        client_idcs[i] = client_idcs[i].astype(np.int)
    return client_idcs


def noniid_sampling(train_labels, alpha, n_clients, seed):
    '''
    Dirichlet distribution
    '''
    np.random.seed(seed)
    n_classes = train_labels.max() + 1
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)

    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]

    client_idcs = [[] for _ in range(n_clients)]

    for c, fracs in zip(class_idcs, label_distribution):

        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):

            client_idcs[i] += [idcs]
    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
    return client_idcs


def trainset_split(true_targets, noise_traget, trainset, client_idcs, num_worker):
    splited_trainset = [Subset(trainset, client_idcs[i]) for i in range(num_worker)]
    noise_client_targets = [np.array(noise_traget)[client_idcs[i]]
                            for i in range(num_worker)]
    true_client_targets = [np.array(true_targets)[client_idcs[i]]
                           for i in range(num_worker)]
    num_client_data = []
    for i in range(len(splited_trainset)):
        num_client_data.append(len(splited_trainset[i]))

    return true_client_targets, noise_client_targets, splited_trainset, num_client_data


def set_output_name(args):
    output_dir_name = 'output_file/dataset_%s/model_%s/optim_%s/loss_%s/iid_%s/noise_%s/R_%s_N_%s_E_%s_B_%s_C_%1.0e_lr_%.1e' % (
        args.dataset, args.model, args.optimizer, args.loss, args.iid, args.noise, args.R, args.num_worker, args.E,
        args.B, args.choosen_fraction, args.lr
    )
    if not args.iid:
        output_dir_name += '_dir_alp_%s' % (args.dir_alpha)
    if args.noise == True:
        output_dir_name += 'type_%s_r_%s' % (args.noise_type, args.noise_ratio)

    if not os.path.exists(output_dir_name):
        os.makedirs(output_dir_name)
    output_dir_name += '/'
    return output_dir_name


def result_dict(purpose, t, result):
    write_dict = OrderedDict()
    write_dict['t'] = t
    write_dict['acc'] = result[0]
    write_dict['top5_acc'] = result[1]
    write_dict['precision'] = result[2]
    write_dict['recall'] = result[3]
    write_dict['f1'] = result[4]
    write_dict['loss'] = result[5]
    file_write(write_dict, purpose)


def file_write(write_dict, purpose):
    output_dir_name = set_output_name(options.args_parser())
    f = open(output_dir_name + purpose + '.txt', 'a')
    if write_dict['t'] == 1:
        d_count = 1
        for k, v in iter(write_dict.items()):
            if d_count < len(write_dict):
                f.write(k + ',')
            else:
                f.write(k + '\n')
            d_count += 1
        d_count = 1
        for k, v in iter(write_dict.items()):
            if d_count < len(write_dict):
                f.write(str(v) + ',')
            else:
                f.write(str(v) + '\n')
            d_count += 1
    elif write_dict['t'] != 1:
        d_count = 1
        for k, v in iter(write_dict.items()):
            if d_count < len(write_dict):
                f.write(str(v) + ',')
            else:
                f.write(str(v) + '\n')
            d_count += 1
    f.close()

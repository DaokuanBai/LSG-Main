import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
import loss_func
from client import Clients
from training import  validate
from datasets import data_loading
from options import args_parser
from utils import trainset_split, iid_sampling, set_random_seed, result_dict, \
    data_noisifying, sample_by_class, n_iid_sampling, Clothing_sampling
import models
import gc


def main(args):

    trainset, testset = data_loading(args)

    ensemble_logits = torch.zeros(len(trainset), args.num_class).cuda()# initialize self-ensemble logits for every sample


    testloader = DataLoader(testset, batch_size=args.B, shuffle=True,num_workers=4)

    true_targets = copy.deepcopy(trainset.targets)

    noise_traget = copy.deepcopy(trainset.targets)

    class_idcs, class_idcs_s = sample_by_class(true_targets, args.num_class)

    if args.noise:

        noise_count = 0

        data_noisifying(args, noise_traget, trainset.targets, class_idcs, trainset.data)

        for i in range(len(trainset.targets)):

            if trainset.targets[i] != true_targets[i]:
                noise_count += 1

        noise_ratio = noise_count / len(trainset.targets)

        print('System noise level:{}'.format(noise_ratio))

    if args.iid and args.dataset != 'clothing1m':
        client_idcs = iid_sampling(class_idcs_s,len(trainset), args.num_worker, args.seed)


    elif not args.iid and args.dataset != 'clothing1m':
        client_idcs = n_iid_sampling(class_idcs_s, len(trainset), args.num_worker, args.seed)

    else:
        client_idcs = Clothing_sampling(len(trainset), args.num_worker)

    true_client_targets, noise_client_traget, split_trainset, num_data = trainset_split(true_targets,noise_traget,trainset, client_idcs, args.num_worker)

    if args.noise:
        client_noise_level = []

        for c in range(args.num_worker):
            noise_count = 0

            for d in range(num_data[c]):
                if noise_client_traget[c][d] != true_client_targets[c][d]:
                    noise_count = noise_count + 1

            client_noise_level.append(noise_count/num_data[c])
            print('client :{:d} noise level:{:.2f} '.format(c,client_noise_level[c]))

    clients = [Clients(args, split_trainset[i]) for i in range(args.num_worker)]

    device = torch.device('cuda:0')



    if args.model == 'CNN1':
        model = models.CNN1() # if you changed the datasets, do not forget to modify the input channel

    elif args.model == 'CNN2':
        model = models.CNN2()

    elif args.model == 'resnet18':
        model = models.resnet18()

    elif args.model == 'resnet50':
        model = models.resnet50(pretrained=True)

    else:
        model = None

    if args.criterion == 'CE':
        criterion = torch.nn.CrossEntropyLoss()

    else:
        criterion = None

    loss_fn_1 = loss_func.cross_entropy()



    global_dict = copy.deepcopy(model.state_dict())

    k = int(args.num_worker * args.choosen_fraction)

    for t in range(args.R):

        total_data = 0

        print('===============The {:d}-th round==============='.format(t + 1))

        selected = np.random.choice(args.num_worker, k, replace=False)

        print('Chosen clients:{}'.format(selected))

        return_dict = {}

        print('--------')

        for i in range(k):

            if args.noise:
                print('client {:d} noise level is {:.3f}'.format(selected[i],client_noise_level[selected[i]]))

            else:
                print('client {:d}'.format(selected[i]))

            clients[selected[i]].train(args,ensemble_logits, model, loss_fn_1, device, return_dict, selected[i], t,testloader,criterion)

            print('--------')

            model.load_state_dict(global_dict)

            total_data += num_data[selected[i]]

        if args.rule == 'fedavg':

            print("Using fedAVG aggregation algorithm........")
            alpha = [] #weights for aggregation

            for i in range(len(selected)):
                alpha.append(num_data[selected[i]] / total_data)

            weighted_update = dict()

            for name, data in global_dict.items():
                weighted_update[name] = torch.zeros_like(data).float()
            # global_dict = self.weights_dict

            for name, data in global_dict.items():

                for i in range(len(return_dict)):  # calculate average weight/bias --> avg_w/b

                    weighted_update[name] += return_dict[str(selected[i])][name] * alpha[i]

                global_dict[name] = data + weighted_update[name]

        model.load_state_dict(global_dict)

        global_acc = validate(testloader, model, criterion, device)

        print('global model acc:{:.4f}'.format(global_acc[0]))

        print('global model top5 acc:{:.4f}'.format(global_acc[1]))

        result_dict('global_acc', t + 1, global_acc)






if __name__ == '__main__':
    args = args_parser()
    gc.collect()
    torch.cuda.empty_cache()
    set_random_seed(args.seed)
    main(args)
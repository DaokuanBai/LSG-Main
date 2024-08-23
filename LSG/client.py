from copy import deepcopy
import torch
from torch.utils.data import DataLoader
import warnings
from training import run_step, validate

warnings.filterwarnings("ignore")


class Clients:
    def __init__(self, args, trainset):
        self.trainset = trainset

        self.ensemble_targets = torch.zeros(len(self.trainset), 10)

        self.count = 0

    def train(self, args, ensemble_logits, model, loss_fn, device, local_dicts, idx, r, testloader, criterion):

        num_data = len(self.trainset)
        trainloader = DataLoader(dataset=self.trainset, batch_size=args.B, shuffle=True, num_workers=4)
        global_dict = deepcopy(model.state_dict())
        if args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

        elif args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
        else:
            optimizer = None

        for i in range(args.E):

            cali = (1 - 0.9 ** (self.count * args.E + i))  #calibrate the startup bias by dividing by cali

            run_step(args, trainloader, model, loss_fn, optimizer, device, num_data, r, ensemble_logits, cali)

            # if (i + 1) % 20 == 0:
            #     global_acc = validate(testloader, model, criterion, device)
            #     print('global model acc:{:.4f}'.format(global_acc[0]))
            #     print('global model top5 acc:{:.4f}'.format(global_acc[1]))

        self.count = self.count + 1

        local_dict = deepcopy(model.state_dict())

        update = dict()

        for k, v in global_dict.items():
            v = v.to(torch.device('cpu'))
            local_dict[k] = local_dict[k].to(torch.device('cpu'))
            update[k] = local_dict[k] - v

        local_dicts[str(idx)] = update

        print('has been selected  for {:d} round'.format(self.count))









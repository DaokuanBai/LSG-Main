import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # FL settings
    parser.add_argument('--num_worker', type=int, default=100,
                        help='number of workers')
    parser.add_argument('--choosen_fraction', type=float, default=0.02,
                        help='fraction of clients selected every round')
    parser.add_argument('--B', type=int, default=128,
                        help='batch size')
    parser.add_argument('--R', type=int, default=600)
    parser.add_argument('--E', type=int, default=1)
    parser.add_argument('--lr', type=int, default=0.001)
    parser.add_argument('--dir_alpha', type=float, default=0.9,
                        help='dirichlet distribution')
    parser.add_argument('--iid', type=bool, default=True,
                        help='data distribution')
    parser.add_argument('--seed', type=int, default=66666,
                        help='random seed')
    parser.add_argument('--rule', type=str, default='fedavg')
    # dataset setiing
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--num_class', type=int, default=10)

    # training setting
    parser.add_argument('--model', type=str, default='CNN1')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='optimizer = {SGD,Adam}')
    parser.add_argument('--criterion', type=str, default='CE')
    parser.add_argument('--loss', type=str, default='CE', help='{CE, SCE}')
    parser.add_argument('--ensemble_label', type=bool, default=True)

    # noise setting
    parser.add_argument('--noise', type=bool, default=True)
    parser.add_argument('--noise_ratio', type=float, default=0.4)
    parser.add_argument('--noise_type', type=str, default='symmetric')
    parser.add_argument('--global_t_w', type=int, default=40)
    parser.add_argument('--gamma', type=float, default=1.0)

    return parser.parse_args()
import torch.nn as nn
import torch.nn.functional as F
def loss_fn_(loss):
    criterion = nn.CrossEntropyLoss()
    if loss == 'CE':
        def loss_fn(output, target):
            loss_ = criterion(output, target)
            return loss_

    elif loss == 'SCE':
        def loss_fn(output, target):
            loss_ = criterion(output, target) + criterion(target, output)
            return loss_
    else:
        pass
    return loss_fn

def cross_entropy():
    def loss_fn(output, target):
            loss_ = F.cross_entropy(output, target)
            return loss_

    return loss_fn



def L1loss( predict, target):
    """
    :param predict: model prediction for original data
    :param target: model prediction for mildly augmented data
    :return: loss
    """
    loss_f = nn.L1Loss(reduction='mean')
    l1_loss = loss_f(predict, target)
    return l1_loss

def L2loss(predict, target):
    """
    :param predict: model prediction for original data
    :param target: model prediction for mildly augmented data
    :return: loss
    """
    loss_f_mse = nn.MSELoss(reduction='mean')
    loss_mse = loss_f_mse(predict, target)
    return loss_mse

def js(p_output, q_output):
    """
    :param predict: model prediction for original data
    :param target: model prediction for mildly augmented data
    :return: loss
    """
    KLDivLoss = nn.KLDivLoss(reduction='mean')
    log_mean_output = ((p_output + q_output) / 2).log()
    return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output)) / 2


def kl_div(input, target):
    KLDivLoss = nn.KLDivLoss(reduction='mean')
    return KLDivLoss(input,target)
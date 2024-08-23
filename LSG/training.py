import torch
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import loss_func


def run_step(args, trainloader, model, loss_fn, optimizer, device, num_data, global_round, ensemble_logits, cali):
    print(num_data)
    model.to(device)
    epoch_loss, total_preds, correct_preds, avg_loss, acc = 0, 0, 0, 0.0, 0.0
    description = "training (the {:d}-batch): tra_Loss = {:.4f} tra_Accuracy = {:.2f}%"
    ensemble_logits = ensemble_logits.to(device)
    with tqdm(trainloader) as epochs:
        for idx, (data, label, id) in enumerate(epochs):

            if cali == 0:
                h_logits = ensemble_logits[id.numpy()]
            else:
                h_logits = ensemble_logits[id.numpy()] / cali

            epochs.set_description(description.format(idx + 1, avg_loss, acc))

            model.train()

            optimizer.zero_grad()

            # for i in range(len(data)):
            #     data[i] = tfs.RandomRotation(30)(data[i])

            data, label = data.to(device), label.to(device)
            # data = data.view(128, 28, 28, 1)

            label = label.long()
            data = data.float()
            output = model(data)

            p = torch.softmax(output, dim=1)
            p_ = p ** (2)
            ps = p_ / p_.sum(dim=1, keepdim=True)

            loss1 = loss_fn(ps, label)

            betaa = args.gamma
            if (global_round < args.global_t_w):
                betaa = args.gamma * global_round / args.global_t_w

            logits1 = torch.softmax(output * 3, dim=1)
            logits2 = torch.softmax(h_logits * 3, dim=1)

            logits1, logits2 = torch.clamp(logits1, min=1e-6, max=1.0), torch.clamp(logits2, min=1e-6, max=1.0)
            loss2 = loss_func.kl_div(logits1, logits2)

            loss = loss1 + betaa * loss2
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

            loss1.backward()
            optimizer.step()

            epoch_loss += loss
            avg_loss = epoch_loss / (idx + 1)
            pred = output.argmax(dim=1)
            correct_preds += pred.eq(label.view_as(pred)).sum()
            acc = correct_preds / num_data * 100
            output = output.detach()
            ensemble_logits[id.numpy()] = 0.1 * output + 0.9 * ensemble_logits[id.numpy()]


def validate(loader, model, criterion, device):
    total_loss = 0.0
    correct = 0.0
    top5_correct = 0.0
    total = 0.0
    accuracy = 0.0
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    with torch.no_grad():
        for idx, (inputs, targets, id) in enumerate(loader):
            model.eval()
            model.to(device)
            inputs = inputs.to(device)
            targets = targets.to(device)
            targets = targets.long()
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            # softmax_func = torch.nn.Softmax(dim=1)
            # soft_output = softmax_func(outputs)
            correct += preds.eq(targets.view_as(preds)).sum()
            total_loss += criterion(outputs, targets)
            accuracy += accuracy_score(targets.cpu().data, preds.cpu().data)
            precision += precision_score(targets.cpu().data, preds.cpu().data, average='macro')
            recall += recall_score(targets.cpu().data, preds.cpu().data, average='macro')
            f1 += f1_score(targets.cpu().data, preds.cpu().data, average='macro')
            maxk = max((1, 5))
            y_resize = targets.view(-1, 1)
            _, top_pred = outputs.topk(maxk, 1, True, True)
            top5_correct += torch.eq(top_pred, y_resize).sum().float().item()
            total = total + len(inputs)

        avg_loss = total_loss / (idx + 1)
        # acc = correct / len(node.test_data.dataset) * 100
        acc = accuracy / (idx + 1) * 100
        top5_acc = top5_correct / total * 100
        precision = precision / (idx + 1) * 100
        recall = recall / (idx + 1) * 100
        f1 = f1 / (idx + 1) * 100
        return acc, top5_acc, precision, recall, f1, avg_loss
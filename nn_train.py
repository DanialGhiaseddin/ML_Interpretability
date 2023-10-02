# New analysis
from data.datasets import TorchDataset
from nn_models import TorchFullyConnected, TorchAttentionBase
import torch
import torch.optim as optim
from torch.autograd import Variable
import torchmetrics
from sklearn.utils import class_weight
import numpy as np

# Dataloader
train_set = TorchDataset('data/Mean_data2.csv', is_train=True, normalize=True, class_weights=[0.2, 0.8], verbose=1)
test_set = TorchDataset('data/Mean_data2.csv', is_train=False, normalize=True, verbose=1)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

# Model Values
output_classes = 2

model = TorchFullyConnected(73, output_classes)
# model = TorchAttentionBase(embed_dim=4, num_heads=2)

# optimizer
optimizer = optim.Adam(model.parameters(), weight_decay=10 ** -4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

if output_classes == 1:
    criterion = torch.nn.BCELoss()
else:
    criterion = torch.nn.CrossEntropyLoss()


metrics = dict()
metrics['accuracy'] = torchmetrics.Accuracy(compute_on_step=False)
# metrics['average_precision'] = torchmetrics.AveragePrecision(pos_label=1, average=None, compute_on_step=False)
# metrics['auc'] = torchmetrics.AUROC(compute_on_step=False, num_classes=2, pos_label=1)
metrics['confusion_matrix'] = torchmetrics.ConfusionMatrix(compute_on_step=False, num_classes=2)


def print_log(metrics_dict, result_list, epoch_n, is_train=True):
    if is_train:
        temp_str = f"Epoch {epoch_n} : \n"
    else:
        temp_str = ""
    for idx, name in enumerate(metrics_dict.keys()):
        if name == 'confusion_matrix':
            # temp_str += f"\t {name}:{result_list[idx][1,0]} \n"
            # temp_str += f"{name}:{result_list[idx]} \n"
            false_neg_rate = result_list[idx][1, 0] / (result_list[idx][1, 0] + result_list[idx][1, 1])
            temp_str += f"\t false_neg_rate: {false_neg_rate} \n"
        else:
            temp_str += f"\t {name}:{result_list[idx]} \n"

    print(temp_str)


best_acc = 0.0

for epoch in range(500):
    train_loss = 0
    train_corrects = 0

    for metric in metrics.values():
        metric.reset()

    for iteration, (x, y, weights) in enumerate(train_loader):
        optimizer.zero_grad()
        x = Variable(x)
        logit = model(x)
        logit = torch.squeeze(logit)
        # loss = criterion(logit, y, weights=weights)
        if output_classes == 1:
            loss = torch.nn.BCELoss(weight=weights)(logit, y)
        else:
            class_weights = class_weight.compute_class_weight(
                class_weight="balanced",
                classes=np.unique(y),
                y=y.numpy()
            )
            class_weights = torch.tensor(class_weights, dtype=torch.float)

            if len(class_weights.shape) < 2:
                class_weights = torch.tensor([1.0, 1.0])

            y = y.type(torch.long)
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
            loss = criterion(logit, y)

            # try:
            #     loss = criterion(logit, y)
            # except:
            #     print("Wait here")

        loss.backward()
        train_loss += loss.item()

        optimizer.step()

        for metric in metrics.values():
            metric(logit, y.type(torch.IntTensor))

    # train_acc = metric.compute()
    train_results = []
    for metric in metrics.values():
        result = metric.compute()
        train_results.append(result)

    # print_log(metrics, train_results, epoch)
    print(train_loss)

    # Validation
    for metric in metrics.values():
        metric.reset()
    test_loss = 0
    for x, y, weights in test_loader:
        x = Variable(x)
        logit = model(x)
        logit = torch.squeeze(logit)

        if output_classes == 1:
            loss = torch.nn.BCELoss(weight=weights)(logit, y)
        else:
            class_weights = class_weight.compute_class_weight(
                class_weight="balanced",
                classes=np.unique(y),
                y=y.numpy()
            )
            class_weights = torch.tensor(class_weights, dtype=torch.float)

            if len(class_weights.shape) < 2:
                class_weights = torch.tensor([1.0, 1.0])

            y = y.type(torch.long)
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
            loss = criterion(logit, y)

        test_loss += loss.item()

        # metric(logit, y.type(torch.IntTensor))
        # average_precision(logit, y)
        for metric in metrics.values():
            metric(logit, y.type(torch.IntTensor))

    # test_acc = metric.compute()
    test_results = []
    for metric in metrics.values():
        result = metric.compute()
        test_results.append(result)

    # print_log(metrics, test_results, epoch, is_train=False)

    if test_results[0] > best_acc:
        # print_log(metrics, test_results, epoch)
        best_acc = test_results[0]
        torch.save(model.state_dict(), 'checkpoints/visualizable.pth')

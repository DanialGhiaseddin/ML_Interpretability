"""
Load the trained model
Compute the metrics for model
Visualize the network with different approaches
  0. Visualize input data
  1. Saliency Map
  2. Use Attention Based Model
  3. Weights averaging and visualizing
  4. Search for interpretability algorithms
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

# New analysis
from data.datasets import TorchDataset
from nn_gradients import VanillaGrad, SmoothGrad
from nn_models import TorchFullyConnected

sns.set_theme()

# Compute vanilla gradient


# Dataloader
test_set = TorchDataset('data/Mean_data2.csv', is_train=False, normalize=True, verbose=1, force_override=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=295, shuffle=False)
x, y, weights = next(iter(test_loader))
x.requires_grad_()
print(torch.mean(x))
# loading and model

model = TorchFullyConnected(73, 2)
model.load_state_dict(torch.load('checkpoints/visualizable.pth'))
model.eval()

temp = x[:, 71]
change_value = torch.std(temp)
modified_x = x.clone()
modified_x[:, 71] = modified_x[:, 71] + change_value / 2

logits1 = model(x)
logits2 = model(modified_x)

ttt = logits2 - logits1

print(torch.mean(ttt, dim=0))

vanilla_grad = VanillaGrad(
    pretrained_model=model, cuda=False)
smooth_grad = SmoothGrad(
    pretrained_model=model, cuda=False)
# vanilla_saliency = vanilla_grad(x)
smooth_saliency = smooth_grad(x, index=0)
# Basic evaluation
'''
metrics = dict()
metrics['accuracy'] = torchmetrics.Accuracy(compute_on_step=False)
metrics['average_precision'] = torchmetrics.AveragePrecision(pos_label=1, average=None, compute_on_step=False)
metrics['auc'] = torchmetrics.AUROC(compute_on_step=False, num_classes=2, pos_label=1)
metrics['confusion_matrix'] = torchmetrics.ConfusionMatrix(compute_on_step=False, num_classes=2)

model.eval()
logit = model(x)
logit = torch.squeeze(logit)
scaler = torch.mean(logit)
loss = torch.nn.BCELoss(weight=weights)(logit, y)
loss.backward(retain_graph=True)

for metric in metrics.keys():
    metric_function = metrics[metric]
    metric_function(logit, y.type(torch.IntTensor))
    result = metric_function.compute()
    print(f'{metric}:', result)
'''
# Saliency Map
# saliency_map = x.grad.data.abs()
saliency_map = smooth_saliency
# saliency_map = vanilla_saliency
#
'''
for ind in range(saliency_map.shape[0]):
    saliency_map[ind] = saliency_map[ind] - np.min(saliency_map[ind])
    saliency_map[ind] = saliency_map[ind] / np.max(saliency_map[ind])
'''
# print(torch.min(saliency_map[ind]))

# print(torch.max(saliency_map[ind]))

# saliency_map = torch.mean(saliency_map, dim=0)
ax = sns.heatmap(saliency_map)
plt.show()

x_mean = torch.mean(x, dim=0)
saliency_map_mean = np.mean(saliency_map, axis=0)

bests = np.argsort(saliency_map_mean)[-5:]

# index, maxes = torch.max(saliency_map_mean)
for i in range(5):
    print(bests[4 - i], ":", test_set.get_column_name(bests[4 - i]))

plt.plot(saliency_map_mean)
plt.show()

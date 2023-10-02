from data.datasets import TorchDataset
import torch
from torchvision import transforms
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import random
import numpy as np
from nn_models import Autoencoder, DenseAutoencoder
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

random.seed(100)
np.random.seed(100)
torch.manual_seed(100)

__title__ = "Autoencoder"
__author__ = "M & K"
__version__ = "0.0.1"

drp_list = ["sputum", "BPMAX", "BPMIN", "Hemoptysis", "SoreThroat",
            "Vomit", "smoker", "addiction", "liver",
            "bodyPain", "Diarrhea", "BS_0",
            "airwayDisease", "ShortnessOfBreath", "corticosteroid",
            "stomachache"]

# train_dataset = TorchDataset('data/Mean_data2.csv', is_train=True, normalize=True, test_split=0.1,
#                              ae_version=True, single_label=0,
#                              drop_list=drp_list, verbose=1, force_override=True)
#
#
# valid_dataset = TorchDataset('data/Mean_data2.csv', is_train=False, normalize=True, test_split=0.1,
#                              ae_version=True, single_label=0,
#                              drop_list=drp_list, verbose=1)

# train_dataset = TorchDataset('data/Mean_data2.csv', is_train=True, normalize=True, test_split=0.1,
#                                     ae_version=True, single_label=1,
#                                     drop_list=drp_list, verbose=1)

# valid_dataset = TorchDataset('data/Mean_data2.csv', is_train=False, normalize=True, test_split=0.1,
#                                     ae_version=True, single_label=1,
#                                     drop_list=drp_list, verbose=1)

train_dataset = TorchDataset('data/Mean_data2.csv', is_train=True, normalize=True, test_split=0.1,
                             ae_version=True,
                             resample_pos=500,
                             force_override=True,
                             drop_list=drp_list, verbose=1)

valid_dataset = TorchDataset('data/Mean_data2.csv', is_train=False, normalize=True, test_split=0.1,
                             ae_version=True,
                             resample_pos=500,
                             drop_list=drp_list, verbose=1)

transforms = transforms.Compose([
    transforms.ToTensor()
])

output_path = './outputs'

num_epochs = 10
batch_size = 8
learning_rate = 1e-4

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

model = DenseAutoencoder(train_dataset.feature_size)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=10 ** -3)

min_valid_loss = np.inf
best_loss = 0
loss = 0

tolerance = 10 ** -3

for epoch in range(num_epochs):
    train_loss = 0.0
    model.train()
    for iteration, (x, y, weights) in enumerate(train_dataloader):
        x = Variable(x)
        # ===================forward=====================
        output, reps = model(x)
        loss = criterion(output, x)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()  # Loss
    train_loss = train_loss / len(train_dataset)
    # ================ Validation =====================
    valid_loss = 0.0
    model.eval()
    for iteration, (x, y, weights) in enumerate(valid_dataloader):
        x = Variable(x)
        target, _ = model(x)
        loss = criterion(target, x)
        valid_loss += loss.item()
    valid_loss = valid_loss / len(valid_dataset)
    # ===================log========================
    print(f'epoch {epoch + 1}/{num_epochs}, loss:{train_loss:.4f}, val_loss:{valid_loss:.4f}')

    if min_valid_loss - valid_loss > tolerance:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss
        # Saving State Dict
        torch.save(model.state_dict(), 'saved_model_alv.pth')

# model = Autoencoder()
# model.load_state_dict(torch.load('saved_model_alv.pth'))
model.eval()
# torch.save(model.state_dict(), './sim_autoencoder.pth')

# %% Get Model Latents
all_data = valid_dataset.x
_, all_latents = model(torch.Tensor(all_data))
all_latents = all_latents.detach().numpy()

pca = PCA()
sclr = StandardScaler()
scrs = pca.fit_transform(sclr.fit_transform(all_latents))
var = pca.explained_variance_ratio_ * 100

fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(scrs[:, 0], scrs[:, 1], 'o')
fig.savefig(f'{output_path}/2d_pca.png')
plt.close(fig)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(scrs[:, 0], scrs[:, 1], scrs[:, 2])
fig.savefig(f'{output_path}/3d_pca.png')  # save the figure to file
plt.close(fig)

tsne = TSNE(n_components=3, n_iter=5000, perplexity=40,
            n_iter_without_progress=1000, random_state=101, method='exact')
sclr = StandardScaler()
t_scrs = tsne.fit_transform(sclr.fit_transform(all_latents))

fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(t_scrs[:, 0], t_scrs[:, 1], 'o')
fig.savefig(f'{output_path}/2d_tsne.png')
plt.close(fig)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(t_scrs[:, 0], t_scrs[:, 1], t_scrs[:, 2])
fig.savefig(f'{output_path}/3d_tsne.png')  # save the figure to file
plt.close(fig)

# # %% For classifying based on error output
# x_train_fin = np.hstack((x_train_fin, y_train_fin.reshape(-1, 1)))
# x_val_fin = np.hstack((x_val_fin, y_val_fin.reshape(-1, 1)))
# # %% Triplet loss

# y_train_pos = np.where(y_train_fin == 1)[0]
# # perc = 1
# # y_train_pos = np.random.choice(y_train_pos,
# #                               size=int(np.round(perc*y_train_pos.shape[0])),
# #                              replace=False)
#
# y_train_neg = np.where(y_train_fin != 1)[0]
# # y_train_neg = np.random.choice(y_train_neg,
# #                               size=int(np.round(perc*y_train_neg.shape[0])),
# #                               replace=False)
#
# y_val_pos = np.where(y_val_fin == 1)[0]
# # y_val_pos = np.random.choice(y_val_pos,
# #                             size=int(np.round(perc*y_val_pos.shape[0])),
# #                            replace=False)
#
# y_val_neg = np.where(y_val_fin != 1)[0]
#
# # y_val_neg = np.random.choice(y_val_neg,
# #                             size=int(np.round(perc*y_val_neg.shape[0])),
# #                            replace=False)
#
# def triplter(x, y):
#
#     y = y.reshape(-1, 1)
#     pos_inds = np.where(y == 1)
#     neg_inds = np.where(y != 1)
#     anch_inds = pos_inds
#
#     x_pos = x[pos_inds[0], :]
#     x_neg = x[neg_inds[0], :]
#     x_anch = x[anch_inds[0], :]
#     return x_anch, x_pos, x_neg
#
# class Dset(torch.utils.data.Dataset):
#     def __init__(self, x, y):
#         'Initialization'
#         self.x = x
#         self.y = y
#
#     def __len__(self):
#         'Denotes the total number of samples'
#         return len(self.x)
#
#     def __getitem__(self, index):
#         'Generates one sample of data'
#         # Select sample
#
#         # Load data and get label
#         X = self.x[index]
#         y = self.y[index]
#
#         return X, y
#
# train_dset = Dset(x_train_fin, y_train_fin)
# val_dset = Dset(x_val_fin, y_val_fin)
#
# num_epochs = 700
# batch_size = 32
# learning_rate = 1e-4
# fet_shap = x_train_fin.shape[1]
# n_trn = x_train_fin.shape[0]
# n_val = x_val_fin.shape[0]
# tolr = 10 ** -3
# dataloader = DataLoader(train_dset, batch_size=128)
# val_dataloader = DataLoader(val_dset, batch_size=128)
#
# # %% Total model
# # !!! 70 30 (40 20) 12
# # !!! Variational auto

# # KL Div
# def kl_div(z_log_var, z_mean):
#     kl_div = -0.5 * torch.sum(1 + z_log_var - z_mean ** 2 - torch.exp(z_log_var),
#                               axis=1)  # sum over latent reps
#     kl_div = kl_div.mean()  # Mean over batch size
#     return kl_div
#
# # %%
#
# if torch.cuda.is_available():
#     tt_model = tt_autoencoder_trip().cuda()
#     print("Cuda loaded")
# else:
#     tt_model = tt_autoencoder_trip().cpu()
#
# criterion_trip = nn.TripletMarginLoss(margin=0.5, reduction='mean')
# criterion = nn.MSELoss(reduction='mean')
# # ADD KL DIVERGANCE
# optimizer = torch.optim.Adam(
#     tt_model.parameters(), lr=learning_rate, weight_decay=10 ** -3)
#
# min_valid_loss = np.inf
# best_loss = 0
# loss = 0
#
# all_train_los_trip = []
# all_val_los_trip = []
# for epoch in range(num_epochs):
#     train_loss = 0.0
#     tt_model.train()
#     for data, labls in dataloader:
#         tmp_dat = data
#
#         tmp_dat = Variable(tmp_dat).cuda()
#         # ===================forward=====================
#         output, x_anch, x_pos, x_neg, reps, z_mean, z_log_var = \
#             tt_model(tmp_dat.float(), labls.float())
#
#         len_pos = x_pos.shape[0]
#         len_neg = x_neg.shape[0]
#         # Choose class negative samples
#         negs = np.random.choice(y_train_neg, size=len_pos, replace=False)
#         tmp_negs = torch.from_numpy(x_train_fin[negs, :]).cuda()
#         poses = np.random.choice(y_train_pos, size=len_pos)
#         tmp_pos = torch.from_numpy(x_train_fin[poses, :]).cuda()
#
#         # ===== for alives =======
#
#         # here pos and neg classes are swaped
#
#         alv_negs = np.random.choice(y_train_pos, size=len_neg, replace=False)
#         alv_poses = np.random.choice(y_train_neg, size=len_neg)
#         tmp_negs_alv = torch.from_numpy(x_train_fin[alv_negs, :]).cuda()
#         tmp_pos_alv = torch.from_numpy(x_train_fin[alv_poses, :]).cuda()
#
#         loss = criterion(output, tmp_dat.float())
#         # loss +=  kl_div(z_log_var, z_mean)
#         if np.isnan(criterion_trip(x_anch, tmp_pos, tmp_negs).item()) or \
#                 np.isnan(criterion_trip(x_neg, tmp_pos_alv, tmp_negs_alv).item()):
#             loss += 0.0
#             loss += criterion_trip(x_neg, tmp_pos_alv, tmp_negs_alv)
#             print("Versiffied")
#         else:
#
#             loss += criterion_trip(x_anch, tmp_pos, tmp_negs)
#             loss += criterion_trip(x_neg, tmp_pos_alv, tmp_negs_alv)
#
#         # ===================backward====================
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()  # Loss
#     train_loss = train_loss / len(dataloader)
#     all_train_los_trip.append(train_loss)
#
#     # ================ Validation =====================
#     valid_loss = 0.0
#     tt_model.eval()
#     for val_data, val_labls in val_dataloader:
#         tmp_dat = val_data.cuda()
#         target, xval_anch, xval_pos, xval_neg, _, z_mean_val, z_log_var_val = \
#             tt_model(tmp_dat.float(),
#                      val_labls.float())
#         len_pos = xval_pos.shape[0]
#         len_neg = xval_neg.shape[0]
#         # Choose class negative samples
#         negs = np.random.choice(y_val_neg, size=len_pos, replace=False)
#         tmp_negs = torch.from_numpy(x_val_fin[negs, :]).cuda()
#         poses = np.random.choice(y_val_pos, size=len_pos, replace=False)
#         tmp_pos = torch.from_numpy(x_val_fin[poses, :]).cuda()
#
#         # ===== for alives =====
#         alv_negs = np.random.choice(y_val_pos, size=len_neg, replace=True)
#         alv_poses = np.random.choice(y_val_neg, size=len_neg)
#         tmp_negs_alv = torch.from_numpy(x_val_fin[alv_negs, :]).cuda()
#         tmp_pos_alv = torch.from_numpy(x_val_fin[alv_poses, :]).cuda()
#
#         val_loss = 0.0
#         val_loss += criterion(target, tmp_dat)
#         # val_loss +=  kl_div(z_log_var_val, z_mean_val)
#         if np.isnan(criterion_trip(xval_anch, tmp_pos, tmp_negs).item()) or \
#                 np.isnan(criterion_trip(xval_neg, tmp_pos_alv, tmp_negs_alv).item()):
#             val_loss += criterion_trip(xval_neg, tmp_pos_alv, tmp_negs_alv)
#             val_loss += 0.0
#         else:
#             val_loss += criterion_trip(xval_anch, tmp_pos, tmp_negs)
#             val_loss += criterion_trip(xval_neg, tmp_pos_alv, tmp_negs_alv)
#
#         valid_loss += val_loss.item()
#     valid_loss = valid_loss / len(val_dataloader)
#     all_val_los_trip.append(valid_loss)
#     # ===================log========================
#     print(f'epoch {epoch + 1}/{num_epochs}, loss:{train_loss:.4f}, val_loss:{valid_loss:.4f}')
#
#     if min_valid_loss - valid_loss > tolr:
#         print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
#         min_valid_loss = valid_loss
#         # Saving State Dict
#         torch.save(tt_model.state_dict(), 'saved_model_tt_tr.pth')
#
# tt_model_trip = tt_autoencoder_trip()
# tt_model_trip.load_state_dict(torch.load('saved_model_tt_tr.pth'))
# tt_model_trip.eval()
#
# plt.figure()
# plt.plot(np.array(all_train_los_trip), 'k-')
# plt.plot(all_val_los_trip, 'r-')
#
# # %%
#
# tt_model_trip.cuda().eval()
# outp, _, _, _, tot_reps_trip, _, _ = tt_model_trip(torch.from_numpy(x).cuda().float(),
#                                                    torch.from_numpy(y).float())
#
# # _, _, _, _, tot_reps_trip, _, _ = tt_model_trip(torch.from_numpy(x_train_fin).cuda().float(),
# #                                               torch.from_numpy(y_train_fin).float())
#
# # , _, _
# tot_reps_trip = tot_reps_trip.detach().cpu().numpy()
#
# tot_reps = tot_reps_trip
# print(tot_reps.shape)
#
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# pca = PCA()
# tsne = TSNE(3)
#
# # tt_scrs = tsne.fit_transform(tot_reps)
#
# tt_scrs = pca.fit_transform(tot_reps)
#
# pca.explained_variance_ratio_ * 100
# plt.figure()
# plt.scatter(tt_scrs[:, 0], tt_scrs[:, 1], c=y)
# plt.xlabel("PC1")
# plt.ylabel("PC1")
# plt.title("PCA results from autoencoder model with MSE and triplet loss")
#
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(tt_scrs[:, 0], tt_scrs[:, 1], tt_scrs[:, 2], c=y)
# plt.title("PCA results from autoencoder model with MSE and triplet loss")
# ax.set_zlabel("PC3")
# ax.set_xlabel("PC1")
# ax.set_ylabel("PC2")
#
# # scipy.io.savemat('AE_trip.mat', {"Trip_scrs":tt_scrs, 'labls': y })
#
# # %% Find delta output-input and train logistic
# from sklearn.metrics import mean_squared_error, classification_report
#
# tt_model_trip.cuda().eval()
#
# outp_train, _, _, _, _, _, _ = tt_model_trip(torch.from_numpy(x_train_fin).cuda().float(),
#                                              torch.from_numpy(y_train_fin).float())
# outp_train = outp_train.detach().cpu().numpy()
# dlt_train = x_train_fin - outp_train
#
# outp_test, _, _, _, _, _, _ = tt_model_trip(torch.from_numpy(x_val_fin).cuda().float(),
#                                             torch.from_numpy(y_val_fin).float())
# outp_test = outp_test.detach().cpu().numpy()
#
# y_pred = []
# errors = np.zeros((x_val_fin.shape[0], 2))
# for patn in range(outp_test.shape[0]):
#     tmp_pat = x_val_fin[patn, :]
#     tmp_pat[-1] = 1.0
#     tmp_y = y_val_fin[patn]
#
#     tmp_out_1, _, _, _, _, _, _ = tt_model_trip(torch.from_numpy(tmp_pat.reshape(1, -1)).cuda().float(),
#                                                 torch.from_numpy(tmp_y.reshape(1)).float())
#     tmp_out_1 = tmp_out_1.detach().cpu().numpy()
#
#     loss_1 = mean_squared_error(tmp_out_1, tmp_pat.reshape(1, -1))
#     errors[patn, 1] = loss_1
#
#     tmp_pat[-1] = 0
#     tmp_out_0, _, _, _, _, _, _ = tt_model_trip(torch.from_numpy(tmp_pat.reshape(1, -1)).cuda().float(),
#                                                 torch.from_numpy(tmp_y.reshape(1)).float())
#     tmp_out_0 = tmp_out_0.detach().cpu().numpy()
#
#     loss_0 = mean_squared_error(tmp_out_0, tmp_pat.reshape(1, -1))
#     errors[patn, 0] = loss_0
#     if loss_1 > loss_0:
#         y_pred.append(0)
#     else:
#         y_pred.append(1)
#
# print(classification_report(y_val_fin, y_pred))
#
# # %% tEMP
#
# new_x = np.hstack((x, dlt_out[:, np.array(abs(dlt_coefs) > 0.01).squeeze()]))
#
# nxtr, nxte, nytr, nyte = train_test_split(new_x, y, stratify=y, test_size=0.2,
#                                           random_state=110)
#
# log_mdl_naiv = LogisticRegression(penalty="l1",
#                                   class_weight="balanced",
#                                   solver="liblinear", max_iter=6000,
#                                   C=2.0)  # Class_weight can ba None to simulate unabalnced state
# # Parameters for bayesian optimisation of log reg
# log_parms = {'C': Real(0.01, 3, 'log-uniform')}
#
# kkkk, log_mdl_dlt = mdl_tester(log_mdl_naiv, log_parms,
#                                xx_train=nxtr, xx_test=nxte,
#                                y_train=nytr, y_test=nyte, scoring='f1_weighted',
#                                cv=5, niter=40,
#                                index=list(range(90)), nam="Logistic")
#
# coef_plot(kkkk, np.arange(90))
#
# naivrf_mdl = RandomForestClassifier(150, max_depth=2, class_weight="balanced",
#                                     max_features="auto", n_jobs=-1, )
#
# rf_parms = {'n_estimators': Integer(100, 1200),
#             'max_depth': Integer(2, 5)}
#
# _, _ = mdl_tester(naivrf_mdl, rf_parms, xx_train=nxtr,
#                   xx_test=nxte,
#                   y_train=nytr, y_test=nyte,
#                   scoring=f2,
#                   cv=5, niter=40, index=dat.columns, nam="RF")
# # %% Find Three units with highest weights from logistic
#
# tt_model_trip.cuda().eval()
#
# _, _, _, _, reps_train, _, _ = tt_model_trip(torch.from_numpy(x_train_fin).cuda().float(),
#                                              torch.from_numpy(y_train_fin).float())
# reps_train = reps_train.detach().cpu().numpy()
#
# _, _, _, _, reps_test, _, _ = tt_model_trip(torch.from_numpy(x_val_fin).cuda().float(),
#                                             torch.from_numpy(y_val_fin).float())
# reps_test = reps_test.detach().cpu().numpy()
#
# log_mdl_naiv = LogisticRegression(penalty="l1",
#                                   class_weight="balanced",
#                                   solver="liblinear", max_iter=6000,
#                                   C=2.0)  # Class_weight can ba None to simulate unabalnced state
# # Parameters for bayesian optimisation of log reg
# log_parms = {'C': Real(0.01, 3, 'log-uniform')
#              }
#
# coefs, _ = mdl_tester(log_mdl_naiv, log_parms,
#                       xx_train=reps_train, xx_test=reps_test,
#                       y_train=y_train_fin, y_test=y_val_fin, scoring=f2,
#                       cv=5, niter=40,
#                       index=list(range(10)), nam="Logistic")  # or average_prec_wei
#
# coef_plot(coefs, np.arange(10))
#
# # %% MI oaf total representations
# tt_model_trip.cuda().eval()
#
# cat_vars = np.array([1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18, 19, 73])
#
# kk = [0]
# kk.extend([8, 9])
#
# kk.extend(list(range(20, 78)))
#
# kk.remove(73)
#
# num_vars = np.array(kk)
#
# all_vars = np.hstack((cat_vars, num_vars))
#
# _, _, _, _, total_reps, _, _ = tt_model_trip(torch.from_numpy(x).cuda().float(),
#                                              torch.from_numpy(y).float())
# total_reps = total_reps.detach().cpu().numpy()
#
# mi_cat = np.zeros((10, len(cat_vars)))
#
# for idx, cati in enumerate(cat_vars):
#     tmp_mi = mutual_info_classif(total_reps, feat_dat.iloc[:, cati])
#     mi_cat[:, idx] = tmp_mi
#
# mi_num = np.zeros((10, len(num_vars)))
#
# for idx, numi in enumerate(num_vars):
#     tmp_mi = mutual_info_regression(total_reps, feat_dat.iloc[:, numi])
#     mi_num[:, idx] = tmp_mi
#
# all_mi = np.hstack((mi_cat, mi_num))
# all_nams = feat_dat.columns[all_vars]
#
# hhh = np.argsort(all_mi[0, :])
# ddd = np.flip(hhh)
#
# plt.figure()
# plt.stem(all_mi[0, ddd])
# plt.xticks(list(range(all_mi.shape[1])), labels=all_nams[ddd], rotation=90)
#
# vvv = all_mi.mean(axis=0)
# hhh = np.argsort(vvv)
# ddd = np.flip(hhh)
#
# plt.figure()
# plt.stem(vvv[ddd])
# plt.xticks(list(range(all_mi.shape[1])), labels=all_nams[ddd], rotation=90)
#
# Mi_pdf = pd.DataFrame(all_mi)  # ), columns= all_nams)
# Mi_pdf["all_inds"] = all_vars
#
# fig, ax = plt.subplots()
# sns.heatmap(all_mi, robust=True, cmap='rocket')
# plt.xticks(list(range(all_mi.shape[1])), labels=all_nams, rotation=90)
#
# plt.show()
#
# x, y = np.meshgrid(range(all_mi.shape[-1]), range(all_mi.shape[0]))
#
# dddd = np.vstack((np.empty((66, 76)), all_mi))
#
# fig, ax = plt.subplots()
# im = ax.hexbin(
#     x.reshape(-1),
#     y.reshape(-1),
#     C=all_mi.reshape(-1),
#     gridsize=20, cmap='rocket'
# )
# fig.colorbar(im, location='right')
# plt.show()
# # the rest of the code is adjustable for best output
# ax.set_aspect(0.8)
# ax.set(xlim=(-4, X.max() + 4,), ylim=(-4, Y.max() + 4))
# ax.axis(False)
# plt.show()
#
# scipy.io.savemat("Hmap.mat", {"mi": all_mi, "all_nams": np.array(all_nams),
#                               "mean_mi": vvv, "mean_id": ddd})
# # %% What the shit raw model does
#
# log_mdl_naiv = LogisticRegression(penalty="l1",
#                                   class_weight="balanced",
#                                   solver="saga", max_iter=6000,
#                                   C=2.0, n_jobs=-1)  # Class_weight can ba None to simulate unabalnced state
# # Parameters for bayesian optimisation of log reg
# log_parms = {'C': Real(0.01, 3, 'log-uniform'),
#              }
#
# log_coefs, log_mdl_raw = mdl_tester(log_mdl_naiv, log_parms,
#                                     xx_train=x_train_fin, xx_test=x_val_fin,
#                                     y_train=y_train_fin, y_test=y_val_fin, scoring="f1_weighted",
#                                     cv=5, niter=40,
#                                     index=feat_dat.columns, nam="Logistic")
#
# plt.figure()
# coef_plot(log_coefs, feat_dat.columns)
#
# scipy.io.savemat("log_weit.mat", {"log_weit": log_coefs, "nams": np.array(feat_dat.columns)})
# # %% Triplet
# from sklearn.decomposition import PCA
# from sklearn.svm import SVC
# from sklearn.metrics import RocCurveDisplay
#
# tt_model_trip.cuda().eval()
#
# _, _, _, _, reps_train, _, _ = tt_model_trip(torch.from_numpy(x_train_fin).cuda().float(),
#                                              torch.from_numpy(y_train_fin).float())
# reps_train = reps_train.detach().cpu().numpy()
#
# _, _, _, _, reps_test, _, _ = tt_model_trip(torch.from_numpy(x_val_fin).cuda().float(),
#                                             torch.from_numpy(y_val_fin).float())
# reps_test = reps_test.detach().cpu().numpy()
#
# # reps_train = big_train_dat
#
# # reps_train, reps_test, y_train_fin, y_val_fin = train_test_split(tot_reps_trip,y,
# #                                                                stratify=y,
# #                                                                test_size=0.2,
# #                                                               random_state=100)
#
# tic_trip = time.perf_counter()
# log_mdl_naiv = LogisticRegression(penalty="elasticnet",
#                                   class_weight="balanced",
#                                   solver="saga", max_iter=6000,
#                                   l1_ratio=0.5, C=2.0,
#                                   n_jobs=-1)  # Class_weight can ba None to simulate unabalnced state
# # Parameters for bayesian optimisation of log reg
# log_parms = {'C': Real(0.01, 3, 'log-uniform'),
#              'l1_ratio': Real(0.01, 1, prior='log-uniform')}
#
# _, log_mdl_trip = mdl_tester(log_mdl_naiv, log_parms,
#                              xx_train=reps_train, xx_test=reps_test,
#                              y_train=y_train_fin, y_test=y_val_fin, scoring=f2,
#                              cv=5, niter=40,
#                              index=dat.columns, nam="Logistic")  # or average_prec_wei
#
# toc_trip = time.perf_counter()
#
# # =================== Fix RF (Wont overfit) ===================
# naivrf_mdl = RandomForestClassifier(150, max_depth=2, class_weight="balanced",
#                                     max_features="auto", n_jobs=-1, )
#
# rf_parms = {'n_estimators': Integer(100, 800),
#             'max_depth': Integer(2, 4)}
#
# _, rf_mdl_trip = mdl_tester(naivrf_mdl, rf_parms, xx_train=reps_train,
#                             xx_test=reps_test,
#                             y_train=y_train_fin, y_test=y_val_fin,
#                             scoring=f2,
#                             cv=5, niter=40, index=dat.columns, nam="RF")
# # roc_ploter(rf_mdl, log_mdl, xx_test=reps_test, y_test=y_test)
#
# # naiv_svc = SVC(C=1,degree=3, probability=True,class_weight="balanced")
# # svc_parms = {"C":Real(0.01, 3, 'log-uniform'), "degree":Integer(2, 5)}
#
# # _, svc_mdl_trip = mdl_tester(naiv_svc, svc_parms, xx_train=reps_train,
# #                              xx_test=reps_test,
# #                             y_train=y_train_fin, y_test=y_val_fin,
# #                             scoring=f2,
# #                             cv=5, niter=40, index=dat.columns, nam="RF")
# # from sklearn.metrics import plot_precision_recall_curve
#
# log_trip_prob = log_mdl_trip.predict_proba(reps_test)[:, 1]
# rf_trip_prob = rf_mdl_trip.predict_proba(reps_test)[:, 1]
# # svc_trip_prob = svc_mdl_trip.predict_proba(reps_test)[:,1]
#
# log_trip_prec, log_trip_rec, _ = precision_recall_curve(y_val_fin, log_trip_prob,
#                                                         pos_label=1)
# rf_trip_prec, rf_trip_rec, _ = precision_recall_curve(y_val_fin, rf_trip_prob,
#                                                       pos_label=1)
# log_trip_fpr, log_trip_tpr, _ = roc_curve(y_val_fin, log_trip_prob)
# rf_trip_fpr, rf_trip_tpr, _ = roc_curve(y_val_fin, rf_trip_prob)
#
# # svc_trip_fpr, svc_trip_tpr,_ = roc_curve(y_val_fin, svc_trip_prob)
#
# # =============== Fix grad (use low depth) ===========
# ensgrd_mdl = GradientBoostingClassifier(max_depth=1,
#                                         n_iter_no_change=30)
#
# ensgrd_parms = {'learning_rate': Real(0.001, 0.01, 'log-uniform'),
#                 'n_estimators': Integer(60, 1000)}
# # 'max_depth': Integer(1,2)}
#
# ensgrd_coefs, ensgrd_mdl_trip = mdl_tester(ensgrd_mdl, ensgrd_parms,
#                                            xx_train=reps_train, xx_test=reps_test,
#                                            y_train=y_train_fin, y_test=y_val_fin,
#                                            scoring=f2,
#                                            cv=5, niter=40,
#                                            index=[], nam="Boosted_gradient")
#
# grd_trip_prob = ensgrd_mdl_trip.predict_proba(reps_test)[:, 1]
#
# grd_trip_fpr, grd_trip_tpr, _ = roc_curve(y_val_fin, grd_trip_prob,
#                                           pos_label=1, drop_intermediate=False)
# fig, ax = plt.subplots()
# RocCurveDisplay.from_estimator(ensgrd_mdl_trip, reps_test, y_val_fin, ax=ax)
# RocCurveDisplay.from_estimator(rf_mdl_trip, reps_test, y_val_fin, ax=ax)
# RocCurveDisplay.from_estimator(log_mdl_trip, reps_test, y_val_fin, ax=ax)
# RocCurveDisplay.from_estimator(log_mdl_raw, x_val_fin, y_val_fin, ax=ax)
# RocCurveDisplay.from_estimator(ensgrd_mdl_raw, x_val_fin, y_val_fin, ax=ax)
#
# fig, ax = plt.subplots()
# PrecisionRecallDisplay.from_estimator(ensgrd_mdl_trip, reps_test, y_val_fin, ax=ax)
# PrecisionRecallDisplay.from_estimator(rf_mdl_trip, reps_test, y_val_fin, ax=ax)
# PrecisionRecallDisplay.from_estimator(log_mdl_trip, reps_test, y_val_fin, ax=ax)
# PrecisionRecallDisplay.from_estimator(log_mdl_raw, x_val_fin, y_val_fin, ax=ax)
# PrecisionRecallDisplay.from_estimator(ensgrd_mdl_raw, x_val_fin, y_val_fin, ax=ax)
#
# # %%
# plt.figure()
# plt.plot(rf_trip_fpr, rf_trip_tpr)
# plt.plot(log_trip_fpr, log_trip_tpr)
# plt.plot(svc_trip_fpr, svc_trip_tpr)
#
# print(roc_auc_score(y_val_fin, log_trip_prob, average='weighted'))
# print(roc_auc_score(y_val_fin, rf_trip_prob, average='weighted'))
# print(roc_auc_score(y_val_fin, svc_trip_prob, average='weighted'))
#
# print(average_precision_score(y_val_fin, log_trip_prob, average='weighted'))
# print(average_precision_score(y_val_fin, rf_trip_prob, average='weighted'))
# print(average_precision_score(y_val_fin, svc_trip_prob, average='weighted'))
#
# # %% Vannila shitty models
#
# tic_raw = time.perf_counter()
#
# log_mdl_naiv = LogisticRegression(penalty="elasticnet",
#                                   class_weight="balanced",
#                                   solver="saga", max_iter=4000,
#                                   l1_ratio=0.5, C=2.0,
#                                   n_jobs=-1)  # Class_weight can ba None to simulate unabalnced state
# # Parameters for bayesian optimisation of log reg
# log_parms = {'C': Real(0.01, 3, 'log-uniform'),
#              'l1_ratio': Real(0.01, 1, prior='log-uniform')}
#
# _, log_mdl_raw = mdl_tester(log_mdl_naiv, log_parms,
#                             xx_train=x_train_fin, xx_test=x_val_fin,
#                             y_train=y_train_fin, y_test=y_val_fin, scoring=f2,
#                             cv=5, niter=40,
#                             index=dat.columns, nam="Logistic")  # or average_prec_wei
#
# toc_raw = time.perf_counter()
# toc_raw - tic_raw
# toc_trip - tic_trip
#
# naivrf_mdl = RandomForestClassifier(150, max_depth=2, class_weight="balanced",
#                                     max_features="auto", n_jobs=-1, )
#
# rf_parms = {'n_estimators': Integer(50, 1500),
#             'max_depth': Integer(2, 6)}
#
# _, rf_mdl_raw = mdl_tester(naivrf_mdl, rf_parms, xx_train=x_train_fin,
#                            xx_test=x_val_fin,
#                            y_train=y_train_fin, y_test=y_val_fin,
#                            scoring=f2,
#                            cv=5, niter=30, index=dat.columns, nam="RF")
#
# log_raw_prob = log_mdl_raw.predict_proba(x_val_fin)[:, 1]
# rf_raw_prob = rf_mdl_raw.predict_proba(x_val_fin)[:, 1]
#
# log_raw_prec, log_raw_rec, _ = precision_recall_curve(y_val_fin, log_raw_prob,
#                                                       pos_label=1)
# rf_raw_prec, rf_raw_rec, _ = precision_recall_curve(y_val_fin, rf_raw_prob,
#                                                     pos_label=1)
# log_raw_fpr, log_raw_tpr, _ = roc_curve(y_val_fin, log_raw_prob)
# rf_raw_fpr, rf_raw_tpr, _ = roc_curve(y_val_fin, rf_raw_prob)
#
# # %% PCA shitty models
# # pc_train, pc_test, pcy_train, pcy_test,_,_ = PCA_cov(dat)  # Pc scores
#
# pca = PCA(n_components=10)
#
# pc_train = pca.fit_transform(x_train_fin)
# pc_test = pca.transform(x_val_fin)
#
# pcy_test = y_val_fin
# pcy_train = y_train_fin
#
# log_mdl_naiv = LogisticRegression(penalty="elasticnet",
#                                   class_weight="balanced",
#                                   solver="saga", max_iter=6000,
#                                   l1_ratio=0.5, C=2.0,
#                                   n_jobs=-1)  # Class_weight can ba None to simulate unabalnced state
# # Parameters for bayesian optimisation of log reg
# log_parms = {'C': Real(0.01, 3, 'log-uniform'),
#              'l1_ratio': Real(0.01, 1, prior='log-uniform')}
#
# _, log_mdl_pca = mdl_tester(log_mdl_naiv, log_parms,
#                             xx_train=pc_train, xx_test=pc_test,
#                             y_train=pcy_train, y_test=pcy_test, scoring=f2,
#                             cv=5, niter=40,
#                             index=dat.columns, nam="Logistic")  # or average_prec_wei
#
# naivrf_mdl = RandomForestClassifier(150, max_depth=2, class_weight="balanced",
#                                     max_features="auto", n_jobs=-1, )
#
# rf_parms = {'n_estimators': Integer(50, 700),
#             'max_depth': Integer(2, 5)}
#
# _, rf_mdl_pca = mdl_tester(naivrf_mdl, rf_parms, xx_train=pc_train,
#                            xx_test=pc_test,
#                            y_train=pcy_train, y_test=pcy_test,
#                            scoring=f2,
#                            cv=5, niter=30, index=dat.columns, nam="RF")
#
# #
# log_pca_prob = log_mdl_pca.predict_proba(pc_test)[:, 1]
# rf_pca_prob = rf_mdl_pca.predict_proba(pc_test)[:, 1]
#
# log_pca_prec, log_pca_rec, _ = precision_recall_curve(pcy_test, log_pca_prob,
#                                                       pos_label=1)
# rf_pca_prec, rf_pca_rec, _ = precision_recall_curve(pcy_test, rf_pca_prob,
#                                                     pos_label=1)
# log_pca_fpr, log_pca_tpr, _ = roc_curve(pcy_test, log_pca_prob)
# rf_pca_fpr, rf_pca_tpr, _ = roc_curve(pcy_test, rf_pca_prob)
#
# # %% plot all
# plt.figure()
# plt.plot(rf_trip_fpr, rf_trip_tpr)
# plt.plot(log_trip_fpr, log_trip_tpr)
# plt.plot(svc_trip_fpr, svc_trip_tpr)
#
# # %% Put everything into a mat file
#
# scipy.io.savemat("perf_pca.mat", {"log_pca_prec": log_pca_prec, "log_pca_rec": log_pca_rec,
#                                   "rf_pca_prec": rf_pca_prec, "rf_pca_rec": rf_pca_rec,
#                                   "log_pca_fpr": log_pca_fpr, "log_pca_tpr": log_pca_tpr,
#                                   "rf_pca_fpr": rf_pca_fpr, "rf_pca_tpr": rf_pca_tpr})
#
# scipy.io.savemat("perf_raw.mat", {"log_raw_prec": log_raw_prec, "log_raw_rec": log_raw_rec,
#                                   "rf_raw_prec": rf_raw_prec, "rf_raw_rec": rf_raw_rec,
#                                   "log_raw_fpr": log_raw_fpr, "log_raw_tpr": log_raw_tpr,
#                                   "rf_raw_fpr": rf_raw_fpr, "rf_raw_tpr": rf_raw_tpr})
#
# scipy.io.savemat("perf_van.mat", {"log_van_prec": log_van_prec, "log_van_rec": log_van_rec,
#                                   "rf_van_prec": rf_van_prec, "rf_van_rec": rf_van_rec,
#                                   "log_van_fpr": log_van_fpr, "log_van_tpr": log_van_tpr,
#                                   "rf_van_fpr": rf_van_fpr, "rf_van_tpr": rf_van_tpr})
#
# scipy.io.savemat("perf_trip.mat", {"log_trip_prec": log_trip_prec, "log_trip_rec": log_trip_rec,
#                                    "rf_trip_prec": rf_trip_prec, "rf_trip_rec": rf_trip_rec,
#                                    "log_trip_fpr": log_trip_fpr, "log_trip_tpr": log_trip_tpr,
#                                    "rf_trip_fpr": rf_trip_fpr, "rf_trip_tpr": rf_trip_tpr})
# # %% Plotting roc & prec recal
#
# plt.figure(figsize=(10, 5))
# roc_ploter(rf_mdl_van, log_mdl_van, xx_test=reps_test_van, y_test=y_test_van)
# roc_ploter(rf_mdl_trip, log_mdl_trip, xx_test=reps_test, y_test=y_test)
# plt.legend(["RF_MSE", "Logistic_MSE",
#             "RF_Triplet", "Logistic_Triplet"])
#
# plt.title("Classification performance of autoencoder representations with and without triplet loss")
#
# fig, ax = plt.subplots(figsize=(12, 7))
#
# PrecisionRecallDisplay.from_estimator(rf_mdl_van, reps_test_van, y_test_van,
#                                       ax=ax, name="RF_MSE")
# PrecisionRecallDisplay.from_estimator(log_mdl_van, reps_test_van, y_test_van,
#                                       ax=ax, name="Logistic_MSE")
#
# PrecisionRecallDisplay.from_estimator(rf_mdl_trip, reps_test, y_test,
#                                       ax=ax, name="RF_Triplet")
# PrecisionRecallDisplay.from_estimator(log_mdl_trip, reps_test, y_test,
#                                       ax=ax, name="Logistic_Triplet")
# plt.title('Precision-Recall curve')
# plt.show()

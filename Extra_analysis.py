# New analysis
from cov_codes.preproc import data_loader, data_spliter, undr_smpl
from cov_codes.RFE_svm import custum_rfe, rfe_chk
from cov_codes.Models_cov import itrmdl_reporter,mdl_tester, coef_plot, permut_imp, nn_itr_mdl, roc_ploter, itr_mdl, nn_bld
from cov_codes.PCA_cov import PCA_cov

from cov_codes.Figures import figure1, figure2, figure3, figure4,\
    figure5,figure6, figure7
dir_nam = "D:\Hadi and Mahdi\Hadi and Mahdi\Locker\CVD\Code ph1\cov_codes\pckg_loader.py"
runfile(dir_nam)
# %% ---------- Data -----------

drp_list = ["sputum", "BPMAX", "BPMIN", "Hemoptysis", "SoreThroat", "Vomit", "smoker", "addiction", "liver",
            "gender", "bodyPain", "Diarrhea", "BS_0", "airwayDisease", "ShortnessOfBreath", "corticosteroid", "stomachache"]
# Inputs: path to the data file, list of features to drop
orig_dat = data_loader("COVID_Data.xlsx", drp_list)
try:
    orig_dat.drop("Unnamed: 0", inplace=True,axis=1)
except:
    None
dat = orig_dat.drop("outcome", axis=1).copy()
x = dat.values
y = orig_dat["outcome"].values

# %% ---------- Plot spearman correlation of features --------------

plt.figure(figsize=(16, 7))
orig_dat.corr(method="spearman")["outcome"].sort_values()[:-1].plot(kind='bar')

# ----------------- data splitting -------------
xx_train, xx_test, y_train, y_test = data_spliter(orig_dat, 0.2)
xx_train_unds, unds_y_train = undr_smpl(
    xx_train, y_train, 800)  # Under sampled
pc_train, pc_test, pcy_train, pcy_test = PCA_cov(orig_dat)  # Pc scores

# ---------- optimisation of one iteration of models ------------
# %% Logistic model
log_mdl_naiv = LogisticRegression(penalty="elasticnet", class_weight="balanced",
                                  solver="saga", max_iter=2000,
                                  l1_ratio=0.5, C=2.0, n_jobs=-1)  # Class_weight can ba None to simulate unabalnced state
# Parameters for bayesian optimisation of log reg
log_parms = {'C': Real(0.01, 3, 'log-uniform'),
             'l1_ratio': Real(0.01, 1, prior='log-uniform')}
# %% Full vanila model
log_coef, log_mdl = mdl_tester(log_mdl_naiv, log_parms, xx_train=xx_train, xx_test=xx_test,
                               y_train=y_train, y_test=y_test, scoring=f2,
                               cv=5, niter=40, index=dat.columns, nam="Logistic")  # or average_prec_wei

# %% Plot coefs of the vanila log model
coef_plot(log_coef, labls=dat.columns)
plt.title("Coefs of logistic model")

# %% Full linear SVM
naiv_svm = SVC(class_weight='balanced', probability=True, kernel='linear')
linsvm_parms = {'C': Real(0.01, 3, 'log-uniform')}
linsvm_coef, linsvm_mdl = mdl_tester(naiv_svm, linsvm_parms, xx_train=xx_train, xx_test=xx_test,
                                     y_train=y_train, y_test=y_test, scoring=f2,
                                     cv=5, niter=40, index=dat.columns, nam="Linear_SVM")

# %% Plot coefs of the linear svm model
coef_plot(linsvm_coef, labls=dat.columns)
plt.title("Coefs of linear SVM")

# %% Non-linear full svm model
svc = SVC(class_weight="balanced", kernel='rbf', probability=True)
svm_parms = {'C': Real(0.01, 2, 'log-uniform'), 'degree': Integer(1,
                                                                  5), 'kernel': Categorical(['poly', 'rbf'])}

# Optimizing SVC
svm_coef, svm = mdl_tester(svc, svm_parms, xx_train=xx_train, xx_test=xx_test,
                           y_train=y_train, y_test=y_test, scoring=f2,
                           cv=5, niter=40, index=dat.columns, nam="RBF_SVM")  # or average_prec_wei

# %% Full Random Forest
naivrf_mdl = RandomForestClassifier(150, max_depth=2, class_weight="balanced",
                                    max_features="auto", n_jobs=-1,)

rf_parms = {'n_estimators': Integer(50, 700),
            'max_depth': Integer(2, 6)}

rf_coefs, rf_mdl = mdl_tester(naivrf_mdl, rf_parms, xx_train=xx_train, xx_test=xx_test,
                              y_train=y_train, y_test=y_test, scoring=f2,
                              cv=5, niter=40, index=dat.columns, nam="RF")

# %% Entropy based feature importance of RF
plt.figure()
coef_plot(coefs=rf_coefs, labls=dat.columns)  # Vanila importances

# %% Permutation importance (based on test set)
permut_imp(mdl=rf_mdl, dat=dat, xx_test=xx_test, y_test=y_test,
           n_repeats=10, scoring="recall_weighted")

# %% Boosted ensemble models
bas_tre = DecisionTreeClassifier(max_depth=4, max_features='auto',
                                 class_weight="balanced")
ensada_mdl = AdaBoostClassifier(bas_tre, n_estimators=100)
ensgrd_mdl = GradientBoostingClassifier(max_depth=2, n_iter_no_change=20)

ensada_parms = {'learning_rate': Real(0.01, 0.3, 'log-uniform'), 'n_estimators': Integer(80, 700),
                }
ensgrd_parms = {'learning_rate': Real(0.01, 0.3, 'log-uniform'), 'n_estimators': Integer(80, 700),
                'max_depth': Integer(2, 6)}

# %% Gradient Boosting vanila full model
ensgrd_coefs, ensgrd_mdl_fit = mdl_tester(ensgrd_mdl, ensgrd_parms, xx_train=xx_train, xx_test=xx_test,
                                          y_train=y_train, y_test=y_test, scoring=f2,
                                          cv=5, niter=40, index=dat.columns, nam="Boosted_gradient")
# %% Permutation feature importance for gradient boosted ensemble
permut_imp(mdl=ensgrd_mdl_fit, dat=dat, xx_test=xx_test,
           y_test=y_test, n_repeats=20, scoring="recall_weighted")

# %% Stacking model
est_lst = [("rf1", RandomForestClassifier(600, class_weight="balanced", max_depth=6)),
           ('grad', GradientBoostingClassifier(learning_rate=0.085,
            max_depth=4, n_iter_no_change=20, n_estimators=400)),
           ('grad1', GradientBoostingClassifier(learning_rate=0.085, max_depth=5, n_iter_no_change=20, n_estimators=400))]

stck_ens = StackingClassifier(est_lst, GradientBoostingClassifier(learning_rate=0.085, max_depth=4, n_iter_no_change=20, n_estimators=400),
                              cv=5, verbose=1, n_jobs=-1)
stck_ens.fit(xx_train, y_train)
stck_y = stck_ens.predict(xx_test)
print(classification_report(y_test, stck_y))

# %% -----------------------------  ANN --------------


nn_mdl = nn_bld(dat=xx_train)
nn_mdl.compile(Adam(10**-4), loss='binary_crossentropy', metrics=[FalseNegatives(), BinaryAccuracy(),
                                                                  AUC()])
cbs = [EarlyStopping(patience=30)]
hist = nn_mdl.fit(xx_train, y_train, batch_size=64, callbacks=cbs, class_weight={0: 1, 1: 4.0},
                  epochs=500,  verbose=1, validation_data=(xx_test, y_test))

# %% Train ANN over multiple iterations to get CIs
nn_auc, nn_f1, nn_f2, nn_prec, nn_rec, nn_ac = nn_itr_mdl(x=x, y=y, niter=10)

# %% -------------- Plot ROC curves of one iteration of models ----------------
roc_ploter(ensgrd_mdl_fit, svm, rf_mdl,
           nn_mdl, y_test=y_test, xx_test=xx_test)


# ----------------------------------------- RFE with linear svm --------------
# %%
naiv_svm = LinearSVC(penalty="l1", C=1, dual=False,
                     class_weight="balanced", verbose=1, max_iter=2000, random_state=101)
linsvm_parms = {'C': Real(0.01, 3, 'log-uniform')}

f1_ful, f2_ful, prec_ful, recal_ful, acu_ful, feats_ful = rfe_chk(
    mdl=naiv_svm, nit=10, dat=dat, mdl_itr=40, parms=linsvm_parms,
    x=x, y=y)
# %% Plot the results of the rfe from the previous cell


def plt_iter(mean_dat=recal_ful.mean(axis=0), std_dat=recal_ful.std(axis=0)/(20**0.5)):
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    x = np.arange(mean_dat.shape[0])
    ax.plot(mean_dat, 'k--')
    ax.fill_between(x, y1=mean_dat, y2=mean_dat+std_dat, color=[0.7, 0.7, 0.7])
    ax.fill_between(x, y1=mean_dat, y2=mean_dat-std_dat, color=[0.7, 0.7, 0.7])


plt_iter()

# ----------------------------- Iteratively train models ----------
# %% Iteratively train models to get mean+set; logistic reg
log_mdl_naiv = LogisticRegression(penalty="elasticnet", class_weight="balanced",
                                  solver="saga", max_iter=2000,
                                  l1_ratio=0.5, C=2.0, n_jobs=-1)
log_auc, log_f1, log_f2, log_prec, log_rec, log_ac = itr_mdl(mdl=log_mdl_naiv, params=log_parms, x=x, y=y, scoring=f2,
                                                             cv=5, niter_bays=40, index=dat.columns, nam="Logistic", niter=10)

itr_scrs = [log_auc, log_f1, log_f2, log_prec, log_rec, log_ac]
nms = ['auc', 'f1', 'f2', 'prec', 'rec', 'acu']
itrmdl_reporter(itr_scrs=itr_scrs, nms=nms)

# %% Iter+linear svm
naiv_svm = SVC(kernel='linear', probability=True, class_weight='balanced')
linsvm_parms = {"C": Real(0.01, 4, prior='log-uniform')}

svm_auc, svm_f1, svm_f2, svm_prec, svm_rec, svm_ac = itr_mdl(mdl=naiv_svm, params=linsvm_parms, x=x, y=y, scoring=f2,
                                                             cv=5, niter_bays=40, index=dat.columns, nam="svm_lin", niter=10)
itr_scrs = [svm_auc, svm_f1, svm_f2, svm_prec, svm_rec, svm_ac]
itrmdl_reporter(itr_scrs=itr_scrs, nms=nms)

# %% Iter+SVC
svc = SVC(class_weight="balanced", kernel='rbf', probability=True)
svm_parms = {'C': Real(0.01, 3, 'log-uniform'), 'degree': Integer(2,
                                                                  5), 'kernel': Categorical(['poly', 'rbf'])}

svc_auc, svc_f1, svc_f2, svc_prec, svc_rec, svc_ac = itr_mdl(mdl=svc, params=svm_parms, x=x, y=y, scoring=f2,
                                                             cv=5, niter_bays=40, index=dat.columns, nam="svm", niter=10)
itr_scrs = [svc_auc, svc_f1, svc_f2, svc_prec, svc_rec, svc_ac]
itrmdl_reporter(itr_scrs=itr_scrs, nms=nms)

# %%Iter + RF
rf_mdl = RandomForestClassifier(150, max_depth=2, class_weight="balanced",
                                max_features="auto", n_jobs=-1)
rf_parms = {'n_estimators': Integer(50, 700),
            'max_depth': Integer(2, 6)}

rf_auc, rf_f1, rf_f2, rf_prec, rf_rec, rf_ac = itr_mdl(mdl=rf_mdl, params=rf_parms, x=x, y=y, scoring=f2,
                                                       cv=5, niter_bays=40, index=dat.columns, nam="rf", niter=10)

itr_scrs = [rf_auc, rf_f1, rf_f2, rf_prec, rf_rec, rf_ac]
itrmdl_reporter(itr_scrs=itr_scrs, nms=nms)
# %%Iter + Boosted ensemble
ensgrd_mdl = GradientBoostingClassifier(max_depth=2, n_iter_no_change=20)

ensgrd_parms = {'learning_rate': Real(0.01, 0.3, 'log-uniform'), 'n_estimators': Integer(80, 700),
                'max_depth': Integer(2, 6)}

grd_auc, grd_f1, grd_f2, grd_prec, grd_rec, grd_ac = itr_mdl(mdl=ensgrd_mdl, params=ensgrd_parms, x=x, y=y, scoring=f2,
                                                             cv=5, niter_bays=40, index=dat.columns, nam="rf", niter=10)

itr_scrs = [grd_auc, grd_f1, grd_f2, grd_prec, grd_rec, grd_ac]
itrmdl_reporter(itr_scrs=itr_scrs, nms=nms)

#%% -------------- Models with PCA ---------------
# ANN
nn_mdl_pca = nn_bld(dat=pc_train)
nn_mdl_pca.compile(Adam(10**-4), loss='binary_crossentropy', metrics=[FalseNegatives(), BinaryAccuracy(),
                                                                      AUC()])
hist_pca = nn_mdl_pca.fit(pc_train, pcy_train, batch_size=64, callbacks=cbs, class_weight={0: 1, 1: 4.0},
                          epochs=500,  verbose=1, validation_data=(pc_test, pcy_test))


# %% RF model with PCA
rf_coefs, rf_mdl_pca = mdl_tester(rf_mdl, rf_parms, nam="RF", niter=40, scoring=f2, cv=5,
                                  xx_train=pc_train, xx_test=pc_test,
                                  index=[],y_train=pcy_train, y_test=pcy_test)

# PCA
svm_coef_pc, svm_pc = mdl_tester(svc, svm_parms, nam="RBF SVM", niter=40, scoring=f2,
                                 xx_train=pc_train,
                                 xx_test=pc_test, y_train=pcy_train,
                                 y_test=pcy_test,index = [])  # or average_prec_wei

# %% log model With pca
_, log_mdl_pc = mdl_tester(log_mdl, parms, niter=40, scoring=f2, cv=5, xx_train=pc_train,
                           xx_test=pc_test, y_train=pcy_train, y_test=pcy_test)  # or average_prec_wei

roc_ploter(nn_mdl_pca, log_mdl_pc, rf_mdl_pca,
           xx_test=pc_test, y_test=pcy_test)


# -------------------------------- Lab group and physician features models -----------
# %%  For plotting rocs of different features with RF model
phys_inds = [9, 71, 47, 30, 0, 27, 28]

cbc = [20, 21, 22, 23, 27, 28, 38, 39, 42, 43,
       44, 45, 46, 50, 51, 59, 60, 65, 66, 67, 68]
non_inv = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 16, 17, 18]
biochm = [24, 25, 26, 29, 33, 34, 35, 37, 55, 56, 69, 70, 71, 72]
bld_gs = [40, 41, 48, 49, 53, 54, 57, 58, 61, 62, 63, 64]

# %%  Use physician features
#
xx_train_phys = xx_train[:, phys_inds]
xx_test_phys = xx_test[:, phys_inds]

log_coef, log_mdl = mdl_tester(log_mdl_naiv, log_parms, xx_train=xx_train_phys,
                               xx_test=xx_test_phys, niter=40, scoring=f2,y_train=y_train,
                               y_test=y_test,index=[])  # or average_prec_wei

# %% linear SVM with physician features
linsvm_coef_phys, linsvm_mdl_phys = mdl_tester(naiv_svm, linsvm_parms, xx_train=xx_train_phys,
                                               xx_test=xx_test_phys, nam="Linear SVM", niter=40, scoring=f2,
                                               index=dat.columns[phys_inds],y_train=y_train,
                                               y_test=y_test)  # or f2

# %% RF model with physicians' features
rfphys_coefs, rfphys_mdl = mdl_tester(rf_mdl, rf_parms, nam="RF", niter=30, scoring=f2,
                                      xx_train=xx_train_phys, xx_test=xx_test_phys,
                                      index=dat.columns[phys_inds],y_train=y_train,
                                      y_test=y_test)

# %% Lab groups
# CBC
_, cbcrf_mdl = mdl_tester(naivrf_mdl, rf_parms, nam="RF", niter=25, scoring=f2, cv=5, xx_train=xx_train[:, cbc],
                          xx_test=xx_test[:, cbc],y_train=y_train,
                          y_test=y_test,index=[])
# %% non-invasive
_, noninvrf_mdl = mdl_tester(naivrf_mdl, rf_parms, nam="RF", niter=25, scoring=f2, cv=5, xx_train=xx_train[:, non_inv],
                             xx_test=xx_test[:, non_inv],y_train=y_train,
                             y_test=y_test,index=[])

# %% biochm
_, biochmrf_mdl = mdl_tester(naivrf_mdl, rf_parms, nam="RF", niter=25, scoring=f2, cv=5, xx_train=xx_train[:, biochm],
                             xx_test=xx_test[:, biochm],y_train=y_train,
                             y_test=y_test,index=[])

# %% bld_gs
_, bld_gsrf_mdl = mdl_tester(naivrf_mdl, rf_parms, nam="RF", niter=25, scoring=f2, cv=5, xx_train=xx_train[:, bld_gs],
                             xx_test=xx_test[:, bld_gs],y_train=y_train,
                             y_test=y_test,index=[])

roc_ploter(cbcrf_mdl, xx_test=xx_test[:, cbc])
roc_ploter(noninvrf_mdl, xx_test=xx_test[:, non_inv])
roc_ploter(biochmrf_mdl, xx_test=xx_test[:, biochm])

roc_ploter(bld_gsrf_mdl, xx_test=xx_test[:, bld_gs])
plt.legend(["CBC", 'Non_invasive', 'Biochemistry', 'Blood Gas'])

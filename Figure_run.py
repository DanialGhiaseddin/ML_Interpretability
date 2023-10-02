# New analysis
from cov_codes.preproc import data_loader, data_spliter, undr_smpl
from cov_codes.RFE_svm import custum_rfe, rfe_chk
from cov_codes.Models_cov import itrmdl_reporter,mdl_tester, coef_plot, permut_imp, nn_itr_mdl, roc_ploter, itr_mdl, nn_bld
from cov_codes.PCA_cov import PCA_cov

from cov_codes.Figures import figure1, figure2, figure3, figure4,\
    figure5,figure6, figure7
# dir_nam = "C:\\Users\\ymerri2\\Dropbox\\Covid paper\\cov_codes\\pckg_loader.py"
# runfile(dir_nam)
# %% ---------- Data -----------

drp_list = ["sputum", "BPMAX", "BPMIN", "Hemoptysis", "SoreThroat", "Vomit", "smoker", "addiction", "liver",
            "gender", "bodyPain", "Diarrhea", "BS_0", "airwayDisease", "ShortnessOfBreath", "corticosteroid", "stomachache"]
# Inputs: path to the data file, list of features to drop
orig_dat = data_loader("data/Mean_data2.csv", drp_list)
try:
    orig_dat.drop("Unnamed: 0", inplace=True,axis=1)
except:
    None
dat = orig_dat.drop("outcome", axis=1).copy()
x = dat.values
y = orig_dat["outcome"].values

#%% Figures
figure1(niter=4,orig_dat=orig_dat)
# figure2(orig_dat=orig_dat)
# figure3(niter=40,orig_dat=orig_dat)
# figure4(bays_itr = 40, nitr = 20, metrc = 3,orig_dat=orig_dat)
# #Metrics: f1 (0) ,f2 (1),precision (2), recall (3), accuracy (4)
# figure5(niter=40,orig_dat=orig_dat)
# figure6(niter=40,orig_dat=orig_dat)
# figure7(niter=40,orig_dat=orig_dat)

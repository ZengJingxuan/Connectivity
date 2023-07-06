import os
import numpy as np
import pandas as pd
import seaborn as sns
import zjx_edited_preprocess as zpr
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list
from scipy.spatial.distance import pdist

old_tar = pd.read_csv(r"D:/Connectivity/metadata/targetneurons.csv")
old_tarN = old_tar.iloc[:, 1]
old_tarNames = np.array(old_tarN)
old_Mt = len(old_tarNames)
old_samples = list(range(1, 8))
old_nsamples = len(old_samples)

old_coarray = np.zeros((old_nsamples, old_Mt, old_Mt))
old_coarray[:, :, :] = np.nan     ##  三维数组

folder_path = 'D:/Connectivity/metadata'
old_partial_name = 'old_tar_cor_sample'
old_files = sorted([file for file in os.listdir(folder_path) if old_partial_name in file])
old_partial_name2 = 'old_tar_name_sample'
old_names = sorted([name for name in os.listdir(folder_path) if old_partial_name2 in name])


for old_samplei in range(old_nsamples):
    old_file = 'old_tar_cor_sample' + str(old_samplei) + '.csv'
    old_name = 'old_tar_name_sample' + str(old_samplei) + '.csv'
    old_file_path = os.path.join(folder_path, old_file)
    old_name_path = os.path.join(folder_path, old_name)
    with open(old_file_path, 'r') as old_cor:
         old_data = pd.read_csv(old_cor, index_col=0, header=None, skiprows=1)     ##  dataframe
    old_data = old_data.to_numpy()
    with open(old_name_path, 'r') as old_Name:
         old_uniqName = pd.read_csv(old_Name, header=None, skiprows=1, usecols=[1])
    old_uniqName = np.array(old_uniqName)
    old_Mu = len(old_uniqName)
    old_cellcs = np.empty(old_Mu)
    old_cellcs = np.array(list(map(np.int_, old_cellcs)))
    for old_celli in range(old_Mu):
        old_indices = np.where(old_tarNames == old_uniqName[old_celli])[0]
        old_cellcs[old_celli] = old_indices[0]
    old_coarray[old_samplei, old_cellcs[:, None], old_cellcs[None, :]] = old_data
print(old_coarray)


old_used_tar_cells = np.where(~np.isnan(old_coarray).all(axis=(0, 1)))[0]
old_used_tar_names = old_tarNames[old_used_tar_cells]
old_Mused_tar = len(old_used_tar_names)
old_used_tar_coarray = old_coarray[:, old_used_tar_cells[:, None], old_used_tar_cells[None, :]]
old_cor_tar_mean = np.nanmean(old_used_tar_coarray, axis=0).squeeze()
### 沿第一纬度计算平均值（即样本间），squeeze删除第一维度

## correlation map
hm_old_cor_tar_mean = sns.clustermap(old_cor_tar_mean, xticklabels=old_used_tar_names, yticklabels=old_used_tar_names, col_cluster=False, row_cluster=False, cmap='jet',vmin=-1, vmax=1,cbar_pos=(0.1, 0.595, 0.03, 0.2))
#
pd.DataFrame(old_cor_tar_mean).to_csv("9DA-mean_old_tar_cor.csv")
pd.DataFrame(old_used_tar_names).to_csv("9DA-mean_old_tar_names.csv")

## histplot
zpr.histplot(old_cor_tar_mean, label='9DA-target')
plt.legend()
plt.show()


############ all neurons ############################################

old_all = pd.read_csv(r"D:/Connectivity/metadata/allneurons.csv")
old_allN = old_all.iloc[:, 1]
old_allNames = np.array(old_allN)
old_Ma = len(old_allNames)
old_allsamples = list(range(1, 8))
old_n_allsamples = len(old_allsamples)

old_allcoarray = np.zeros((old_n_allsamples, old_Ma, old_Ma))
old_allcoarray[:, :, :] = np.nan     ##  三维数组

folder_path = 'D:/Connectivity/metadata'
old_all_partial_name = 'old_cor_sample'
old_all_files = sorted([all_file for all_file in os.listdir(folder_path) if old_all_partial_name in all_file])
old_all_partial_name2 = 'old_name_sample'
old_all_names = sorted([all_name for all_name in os.listdir(folder_path) if old_all_partial_name2 in all_name])
###

for old_allsamplei in range(old_n_allsamples):
    old_all_file = 'old_cor_sample' + str(old_allsamplei) + '.csv'
    old_all_name = 'old_name_sample' + str(old_allsamplei) + '.csv'
    old_all_file_path = os.path.join(folder_path, old_all_file)
    old_all_name_path = os.path.join(folder_path, old_all_name)
    with open(old_all_file_path, 'r') as old_all_cor:
         old_all_data = pd.read_csv(old_all_cor, index_col=0, header=None, skiprows=1)     ##  dataframe
    old_all_data = old_all_data.to_numpy()
    with open(old_all_name_path, 'r') as old_allName:
         old_all_uniqName = pd.read_csv(old_allName, header=None, skiprows=1, usecols=[1])
    old_all_uniqName = np.array(old_all_uniqName)
    old_Mau = len(old_all_uniqName)
    old_allcellcs = np.empty(old_Mau)
    old_allcellcs = np.array(list(map(np.int_, old_allcellcs)))
    for old_acelli in range(old_Mau):
        old_all_indices = np.where(old_allNames == old_all_uniqName[old_acelli])[0]
        old_allcellcs[old_acelli] = old_all_indices[0]
    old_allcoarray[old_allsamplei, old_allcellcs[:, None], old_allcellcs[None, :]] = old_all_data
print(old_allcoarray)



old_used_cells = np.where(~np.isnan(old_allcoarray).all(axis=(0, 1)))[0]
old_used_names = old_allNames[old_used_cells]
old_Mused = len(old_used_names)
old_used_coarray = old_allcoarray[:, old_used_cells[:, None], old_used_cells[None, :]]
old_cor_mean = np.nanmean(old_used_coarray, axis=0).squeeze()
### 在所有sample中只出现一次的neuron也包括，但可能不存在相应的cor,即cor_mean中会出现NaN

old_temp_cor_mean = np.nan_to_num(old_cor_mean)   ## 处理nan,暂变为0，以便聚类
old_Y = pdist(old_temp_cor_mean)
old_Z = linkage(old_Y)
old_Z_tree = optimal_leaf_ordering(old_Z, old_Y)
old_Z_index = leaves_list(old_Z_tree)
old_cor_mean_ordered = old_cor_mean[old_Z_index][:, old_Z_index]
old_used_names_ordered = old_used_names[old_Z_index]
old_used_xnames = old_used_names_ordered.copy()

## correlation map
sns.set(font_scale=0.8)
hm_old_cor_mean = sns.clustermap(old_cor_mean_ordered, xticklabels=True, yticklabels=True, col_cluster=False, row_cluster=False, cmap='jet', vmin=-1, vmax=1, cbar_pos=(0.02, 0.79, 0.03, 0.2), figsize=(6.7, 8))
old_used_ynames = zpr.re_name_mean(old_used_names_ordered)
hm_old_cor_mean.ax_heatmap.set_yticklabels(old_used_ynames)
hm_old_cor_mean.ax_heatmap.set_xticklabels(old_used_xnames)

pd.DataFrame(old_cor_mean_ordered).to_csv("9DA-mean_old_cor.csv")
pd.DataFrame(old_used_xnames).to_csv("9DA-mean_old_names.csv")

## histplot
zpr.histplot(old_cor_mean, label='9DA')
plt.legend()
plt.show()


################ dynamic neurons ######################################################

old_dy = pd.read_csv(r"D:/Connectivity/metadata_dynamics/allneurons.csv")
old_dyN = old_dy.iloc[:, 1]
old_dyNames = np.array(old_dyN)
old_Mdy = len(old_dyNames)
old_dysamples = list(range(1, 8))
old_n_dysamples = len(old_dysamples)

old_dycoarray = np.zeros((old_n_dysamples, old_Mdy, old_Mdy))
old_dycoarray[:, :, :] = np.nan     ##  三维数组

dy_folder_path = 'D:/Connectivity/metadata_dynamics'
old_dy_partial_name = 'old_dy_cor_sample'
old_dy_files = sorted([dy_file for dy_file in os.listdir(dy_folder_path) if old_dy_partial_name in dy_file])
old_dy_partial_name2 = 'old_dy_name_sample'
old_dy_names = sorted([dy_name for dy_name in os.listdir(dy_folder_path) if old_dy_partial_name2 in dy_name])
###

for old_dysamplei in range(old_n_dysamples):
    old_dy_file = 'old_dy_cor_sample' + str(old_dysamplei) + '.csv'
    old_dy_name = 'old_dy_name_sample' + str(old_dysamplei) + '.csv'
    old_dy_file_path = os.path.join(dy_folder_path, old_dy_file)
    old_dy_name_path = os.path.join(dy_folder_path, old_dy_name)
    with open(old_dy_file_path, 'r') as old_dy_cor:
         old_dy_data = pd.read_csv(old_dy_cor, index_col=0, header=None, skiprows=1)     ##  dataframe
    old_dy_data = old_dy_data.to_numpy()
    with open(old_dy_name_path, 'r') as old_dyName:
         old_dy_uniqName = pd.read_csv(old_dyName, header=None, skiprows=1, usecols=[1])
    old_dy_uniqName = np.array(old_dy_uniqName)
    old_Mdyu = len(old_dy_uniqName)
    old_dycellcs = np.empty(old_Mdyu)
    old_dycellcs = np.array(list(map(np.int_, old_dycellcs)))
    for old_dycelli in range(old_Mdyu):
        old_dy_indices = np.where(old_dyNames == old_dy_uniqName[old_dycelli])[0]
        old_dycellcs[old_dycelli] = old_dy_indices[0]
    old_dycoarray[old_dysamplei, old_dycellcs[:, None], old_dycellcs[None, :]] = old_dy_data
print(old_dycoarray)



old_used_dy_cells = np.where(~np.isnan(old_dycoarray).all(axis=(0, 1)))[0]
old_used_dy_names = old_dyNames[old_used_dy_cells]
old_Mdyused = len(old_used_dy_names)
old_used_dy_coarray = old_dycoarray[:, old_used_dy_cells[:, None], old_used_dy_cells[None, :]]
old_dy_mean = np.nanmean(old_used_dy_coarray, axis=0).squeeze()
### 在所有sample中只出现一次的neuron也包括，但可能不存在相应的cor,即cor_mean中会出现NaN

old_temp_dy_mean = np.nan_to_num(old_dy_mean)   ## 处理nan,暂变为0，以便聚类
old_dy_Y = pdist(old_temp_dy_mean)
old_dy_Z = linkage(old_dy_Y)
old_dy_Z_tree = optimal_leaf_ordering(old_dy_Z, old_dy_Y)
old_dy_Z_index = leaves_list(old_dy_Z_tree)
old_dy_mean_ordered = old_dy_mean[old_dy_Z_index][:, old_dy_Z_index]
old_used_dy_names_ordered = old_used_dy_names[old_dy_Z_index]
old_used_dy_xnames = old_used_dy_names_ordered.copy()

## correlation map
sns.set(font_scale=0.8)
hm_old_dy_mean = sns.clustermap(old_dy_mean_ordered, xticklabels=True, yticklabels=True, col_cluster=False, row_cluster=False, cmap='jet', vmin=-1, vmax=1, cbar_pos=(0.02, 0.79, 0.03, 0.2), figsize=(6.7, 8))
old_used_dy_ynames = zpr.re_name_mean(old_used_dy_names_ordered)
hm_old_dy_mean.ax_heatmap.set_yticklabels(old_used_dy_ynames)
hm_old_dy_mean.ax_heatmap.set_xticklabels(old_used_dy_xnames)

pd.DataFrame(old_dy_mean_ordered).to_csv("9DA-mean-F_old_dy_cor.csv")
pd.DataFrame(old_used_dy_xnames).to_csv("9DA-mean-F_old_dy_names.csv")

## histplot
zpr.histplot(old_dy_mean, label='9DA-dynamics')
plt.legend()
plt.show()

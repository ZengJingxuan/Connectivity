import os
import numpy as np
import pandas as pd
import seaborn as sns
import zjx_edited_preprocess as zpr
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list
from scipy.spatial.distance import pdist

tar = pd.read_csv(r"D:/Connectivity/metadata/targetneurons.csv")
tarN = tar.iloc[:, 1]
tarNames = np.array(tarN)
Mt = len(tarNames)
samples = list(range(1, 9))
nsamples = len(samples)

coarray = np.zeros((nsamples, Mt, Mt))
coarray[:, :, :] = np.nan     ##  三维数组

folder_path = 'D:/Connectivity/metadata'
partial_name = 'tar_cor_sample'
files = sorted([file for file in os.listdir(folder_path) if partial_name in file])
partial_name2 = 'tar_name_sample'
names = sorted([name for name in os.listdir(folder_path) if partial_name2 in name])


for samplei in range(nsamples):
    file = 'tar_cor_sample' + str(samplei) + '.csv'
    name = 'tar_name_sample' + str(samplei) + '.csv'
    file_path = os.path.join(folder_path, file)
    name_path = os.path.join(folder_path, name)
    with open(file_path, 'r') as cor:
         data = pd.read_csv(cor, index_col=0, header=None, skiprows=1)     ##  dataframe
    data = data.to_numpy()
    with open(name_path, 'r') as Name:
         uniqName = pd.read_csv(Name, header=None, skiprows=1, usecols=[1])
    uniqName = np.array(uniqName)
    Mu = len(uniqName)
    cellcs = np.empty(Mu)
    cellcs = np.array(list(map(np.int_, cellcs)))
    for celli in range(Mu):
        indices = np.where(tarNames == uniqName[celli])[0]
        cellcs[celli] = indices[0]
    coarray[samplei, cellcs[:, None], cellcs[None, :]] = data
print(coarray)

used_tar_cells = np.where(~np.isnan(coarray).all(axis=(0, 1)))[0]
used_tar_names = tarNames[used_tar_cells]
Mused_tar = len(used_tar_names)
used_tar_coarray = coarray[:, used_tar_cells[:, None], used_tar_cells[None, :]]
cor_tar_mean = np.nanmean(used_tar_coarray, axis=0).squeeze()
### 沿第一纬度计算平均值（即样本间），squeeze删除第一维度

## correlation map
hm_cor_tar_mean = sns.clustermap(cor_tar_mean, xticklabels=used_tar_names, yticklabels=used_tar_names, col_cluster=False, row_cluster=False, cmap='jet',vmin=-1, vmax=1,cbar_pos=(0.1, 0.595, 0.03, 0.2))
#
pd.DataFrame(cor_tar_mean).to_csv("1DA-mean_tar_cor.csv")
pd.DataFrame(used_tar_names).to_csv("1DA-mean_tar_names.csv")


zpr.histplot(cor_tar_mean, label='1DA-target')
plt.legend()
plt.show()


############ all neurons ############################################

all = pd.read_csv(r"D:/Connectivity/metadata/allneurons.csv")
allN = all.iloc[:, 1]
allNames = np.array(allN)
Ma = len(allNames)
allsamples = list(range(1, 9))
n_allsamples = len(allsamples)

allcoarray = np.zeros((n_allsamples, Ma, Ma))
allcoarray[:, :, :] = np.nan     ##  三维数组

folder_path = 'D:/Connectivity/metadata'
all_partial_name = 'cor_sample'
all_files = sorted([all_file for all_file in os.listdir(folder_path) if all_partial_name in all_file])
all_partial_name2 = 'name_sample'
all_names = sorted([all_name for all_name in os.listdir(folder_path) if all_partial_name2 in all_name])
###

for allsamplei in range(n_allsamples):
    all_file = 'cor_sample' + str(allsamplei) + '.csv'
    all_name = 'name_sample' + str(allsamplei) + '.csv'
    all_file_path = os.path.join(folder_path, all_file)
    all_name_path = os.path.join(folder_path, all_name)
    with open(all_file_path, 'r') as all_cor:
         all_data = pd.read_csv(all_cor, index_col=0, header=None, skiprows=1)     ##  dataframe
    all_data = all_data.to_numpy()
    with open(all_name_path, 'r') as allName:
         all_uniqName = pd.read_csv(allName, header=None, skiprows=1, usecols=[1])
    all_uniqName = np.array(all_uniqName)
    Mau = len(all_uniqName)
    allcellcs = np.empty(Mau)
    allcellcs = np.array(list(map(np.int_, allcellcs)))
    for acelli in range(Mau):
        all_indices = np.where(allNames == all_uniqName[acelli])[0]
        allcellcs[acelli] = all_indices[0]
    allcoarray[allsamplei, allcellcs[:, None], allcellcs[None, :]] = all_data
print(allcoarray)



used_cells = np.where(~np.isnan(allcoarray).all(axis=(0, 1)))[0]
used_names = allNames[used_cells]
Mused = len(used_names)
used_coarray = allcoarray[:, used_cells[:, None], used_cells[None, :]]
cor_mean = np.nanmean(used_coarray, axis=0).squeeze()
### 在所有sample中只出现一次的neuron也包括，但可能不存在相应的cor,即cor_mean中会出现NaN

temp_cor_mean = np.nan_to_num(cor_mean)   ## 处理nan,暂变为0，以便聚类
Y = pdist(temp_cor_mean)
Z = linkage(Y)
Z_tree = optimal_leaf_ordering(Z, Y)
Z_index = leaves_list(Z_tree)
cor_mean_ordered = cor_mean[Z_index][:, Z_index]
used_names_ordered = used_names[Z_index]
used_xnames = used_names_ordered.copy()

## correlation map
sns.set(font_scale=0.8)
hm_cor_mean = sns.clustermap(cor_mean_ordered, xticklabels=True, yticklabels=True, col_cluster=False, row_cluster=False, cmap='jet', vmin=-1, vmax=1, cbar_pos=(0.02, 0.79, 0.03, 0.2), figsize=(6.7, 7))
used_ynames = zpr.re_name_mean(used_names_ordered)
hm_cor_mean.ax_heatmap.set_yticklabels(used_ynames)
hm_cor_mean.ax_heatmap.set_xticklabels(used_xnames)

pd.DataFrame(cor_mean_ordered).to_csv("1DA-mean_cor.csv")
pd.DataFrame(used_xnames).to_csv("1DA-mean_names.csv")

## histplot
zpr.histplot(cor_mean, label='1DA')
plt.legend()
plt.show()


################ dynamic neurons ######################################################

dy = pd.read_csv(r"D:/Connectivity/metadata_dynamics/allneurons.csv")
dyN = dy.iloc[:, 1]
dyNames = np.array(dyN)
Mdy = len(dyNames)
dysamples = list(range(1, 9))
n_dysamples = len(dysamples)

dycoarray = np.zeros((n_dysamples, Mdy, Mdy))
dycoarray[:, :, :] = np.nan     ##  三维数组

dy_folder_path = 'D:/Connectivity/metadata_dynamics'
dy_partial_name = 'dy_cor_sample'
dy_files = sorted([dy_file for dy_file in os.listdir(dy_folder_path) if dy_partial_name in dy_file])
dy_partial_name2 = 'dy_name_sample'
dy_names = sorted([dy_name for dy_name in os.listdir(dy_folder_path) if dy_partial_name2 in dy_name])
###

for dysamplei in range(n_dysamples):
    dy_file = 'dy_cor_sample' + str(dysamplei) + '.csv'
    dy_name = 'dy_name_sample' + str(dysamplei) + '.csv'
    dy_file_path = os.path.join(dy_folder_path, dy_file)
    dy_name_path = os.path.join(dy_folder_path, dy_name)
    with open(dy_file_path, 'r') as dy_cor:
         dy_data = pd.read_csv(dy_cor, index_col=0, header=None, skiprows=1)     ##  dataframe
    dy_data = dy_data.to_numpy()
    with open(dy_name_path, 'r') as dyName:
         dy_uniqName = pd.read_csv(dyName, header=None, skiprows=1, usecols=[1])
    dy_uniqName = np.array(dy_uniqName)
    Mdyu = len(dy_uniqName)
    dycellcs = np.empty(Mdyu)
    dycellcs = np.array(list(map(np.int_, dycellcs)))
    for dycelli in range(Mdyu):
        dy_indices = np.where(dyNames == dy_uniqName[dycelli])[0]
        dycellcs[dycelli] = dy_indices[0]
    dycoarray[dysamplei, dycellcs[:, None], dycellcs[None, :]] = dy_data
print(dycoarray)


used_dy_cells = np.where(~np.isnan(dycoarray).all(axis=(0, 1)))[0]
used_dy_names = dyNames[used_dy_cells]
Mdyused = len(used_dy_names)
used_dy_coarray = dycoarray[:, used_dy_cells[:, None], used_dy_cells[None, :]]
dy_mean = np.nanmean(used_dy_coarray, axis=0).squeeze()
### 在所有sample中只出现一次的neuron也包括，但可能不存在相应的cor,即cor_mean中会出现NaN

temp_dy_mean = np.nan_to_num(dy_mean)   ## 处理nan,暂变为0，以便聚类
dy_Y = pdist(temp_dy_mean)
dy_Z = linkage(dy_Y)
dy_Z_tree = optimal_leaf_ordering(dy_Z, dy_Y)
dy_Z_index = leaves_list(dy_Z_tree)
dy_mean_ordered = dy_mean[dy_Z_index][:, dy_Z_index]
used_dy_names_ordered = used_dy_names[dy_Z_index]
used_dy_xnames = used_dy_names_ordered.copy()

## correlation map
sns.set(font_scale=0.8)
hm_dy_mean = sns.clustermap(dy_mean_ordered, xticklabels=True, yticklabels=True, col_cluster=False, row_cluster=False, cmap='jet', vmin=-1, vmax=1, cbar_pos=(0.02, 0.79, 0.03, 0.2), figsize=(6.7, 7))
used_dy_ynames = zpr.re_name_mean(used_dy_names_ordered)
hm_dy_mean.ax_heatmap.set_yticklabels(used_dy_ynames)
hm_dy_mean.ax_heatmap.set_xticklabels(used_dy_xnames)

pd.DataFrame(dy_mean_ordered).to_csv("1DA-mean_dy_cor.csv")
pd.DataFrame(used_dy_xnames).to_csv("1DA-mean_dy_names.csv")

## histplot
zpr.histplot(dy_mean, label='1DA-dynamics_60')
plt.legend()
plt.show()

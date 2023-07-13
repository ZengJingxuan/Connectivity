import pandas as pd
import numpy as np
import matplotlib as mpl
from scipy.signal import medfilt
mpl.use('TkAgg')
from scipy.optimize import curve_fit
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list
from scipy.spatial.distance import pdist
import math
from scipy.stats import ranksums
import os


# for check plots
def monplot(dat, title='', subplot_shape=111):
    nrows = 200     # number of plots to show
    ncols = 1000    # length of plots (volumes)
    print('datasize', dat.shape)
    mpl.use('TkAgg')
    if subplot_shape == 111:
        plt.figure()
        plt.plot(dat[:ncols, :nrows])
        plt.title(title)
        plt.show()
    else:
        plt.subplot(subplot_shape)
        plt.plot(dat[:ncols, :nrows])
        plt.title(title)


# median filter like medfilt1 in matlab with 'omitnan','truncate' options
# mat is a numpy array, size is filter size (can be even)
def medfilt(mat, size):
    assert len(mat.shape) == 2
    assert type(size) is int          # 断言，确保数据类型
    n, m = mat.shape
    pre = math.ceil((size-1)/2)       # 取上整数
    post = math.floor((size-1)/2)     # 不取上
    arr = np.zeros((n, m, size))
    arr[:, :] = np.nan
    arr[:, :, pre] = mat     ### 确保赋值在中心位置
    for shift in range(pre, 0, -1):
        arr[shift:, :, pre - shift] = mat[:-shift, :]
    for shift in range(1, post+1):
        arr[:-shift, :, pre + shift] = mat[shift:, :]
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)  # suppress all nan warnings
    out_mat = np.nanmedian(arr, 2)                   # median along axis=2
    warnings.simplefilter("default")
    return out_mat


# remove outlier in matlab way, ch1: CFP    创建副本避免覆盖
def rep_outl(ch1_med, ch2_med):
    tmp_median = np.median(ch1_med, 0)
    flag_under_median = ch1_med < tmp_median/10     # 布尔型数组 true & false
    ch1_med_2 = ch1_med.copy()
    ch2_med_2 = ch2_med.copy()
    ch1_med_2[flag_under_median] = np.nan
    ch2_med_2[flag_under_median] = np.nan
    ch1_med_3 = medfilt(ch1_med_2, 4)
    ch2_med_3 = medfilt(ch2_med_2, 4)
    ch1_med_4 = ch1_med_2.copy()
    ch2_med_4 = ch2_med_2.copy()
    ch1_med_4[flag_under_median] = ch1_med_3[flag_under_median]
    ch2_med_4[flag_under_median] = ch2_med_3[flag_under_median]
    # median filter only replace flags!! 其他非flag值不变，和ch1_med_2相等（没有重复中值滤波）
    flag_outliers = (np.sum(flag_under_median, 0) > 66) | np.any(np.isnan(ch1_med_4), 0) | np.any(np.isnan(ch2_med_4), 0);
    ### 400 changed to 66, since time series 6000 to 1000, flag_outliers也是布尔型数组
    tcrs = ch2_med_4/ch1_med_4
    tcrs = tcrs[:, ~flag_outliers]      ## slicing outlier列
    return tcrs, flag_outliers


## x: time series, y: data
def rm_nan(x, y):
    dt = list(zip(x, y))
    dt = [n for n in dt if ~np.isnan(n[1])]       # 第二维度
    x, y = zip(*dt)
    return np.asarray(x), np.asarray(y)

def de_trend(mat):
    def fun(x, a, b, c):
        return c * np.exp(-a * x) + b
    mat = mat.T
    time_series = np.linspace(0, mat.shape[1] - 1, num=mat.shape[1])
    new = []
    for data in mat:
    ### x, y = rm_nan(list(time_series), list(data))
        x, y = list(time_series), list(data)
        f, err = curve_fit(fun, x, y, p0=[0.003, 0.5, 0.5], maxfev=5000)
        curve = fun(time_series, f[0], f[1], f[2])
        new.append(curve)
    new = np.asarray(new)
    mat = mat / new
    return mat.T, new.T


def norm(arr):
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    return (arr - mean) / std


def normalize2(ch1_d, ch2_d):

    # denoise by median filter
    ch1_med = medfilt(ch1_d, 7)
    ch2_med = medfilt(ch2_d, 7)
    # check plot
    plt.figure()
    monplot(ch1_d, 'ch1 original', 331)
    monplot(ch1_med, 'ch1 medfilt denoise', 332)
    monplot(ch2_d, 'ch2 original', 333)
    monplot(ch2_med, 'ch2 medfilt denoise', 334)

    # remove outliers
    tcrs_rep, flag_outliers = rep_outl(ch1_med, ch2_med)
    print('outliers:', np.where(flag_outliers)[0])     # outlier index
    monplot(tcrs_rep, 'remove outliers, ch2/ch1', 335)

    # detrend
    tcrs_dt, new = de_trend(tcrs_rep)
    monplot(tcrs_dt, 'detrend, ch2/ch1', 336)

    # normalization
    tcrs_nm = norm(tcrs_dt)
    monplot(tcrs_nm, 'normalized, ch2/ch1', 337)

    # show check plots
    plt.subplots_adjust(hspace=0.5)    # 设置子图间距
    plt.show()

    return tcrs_nm, flag_outliers

def main(ch1_p,ch2_p):
    ch1 = pd.read_csv(ch1_p, index_col=0)
    ch2 = pd.read_csv(ch2_p, index_col=0)
    name = ch1.columns
    ch1_d = ch1.values
    ch2_d = ch2.values
    fret_hm_nm, flag_outliers = normalize2(ch1_d, ch2_d)
    fret = pd.DataFrame(fret_hm_nm, columns=name[~flag_outliers])
    return fret

### select dynamic neurons   mat = norm(time x neuron)
def select(fret, n):
    ind = list(fret.var().sort_values()[-n:].index)
    mat = pd.DataFrame()
    for i in ind:
        mat = pd.concat([mat, fret[i]], axis=1)
    return mat

### select target neurons
def select_tar(ch1_p, ch2_p, arr):
    y1 = main(ch1_p, ch2_p)
    name = y1.columns
    targets = np.array(['AVAL','AVAR','AVDL','AVDR','AVEL','AVER','RIML','RIMR','AIBL','AIBR','RIBL','RIBR','RMEL','RMER','AVBL','AVBR'])
    matches = []
    for query in targets:
      print(query)
      match = np.argwhere(name==query)
      if len(match) > 0:
       print(match)
       matches.append(match[0][0])
    print(matches)
    arr = arr.iloc[:, matches]
    return arr


### matlab way to cluster, re_order all neurons
def cluster(arr, use_optimal_leaf_ordering=True):
    if use_optimal_leaf_ordering:
        dist = pdist(arr.T)
        lin = linkage(dist)
        ordered_tree = optimal_leaf_ordering(lin, dist)
        ordered_index = leaves_list(ordered_tree)
        ordered_arr = arr.iloc[:, ordered_index]
        sns.set(font_scale=0.8)
        hm = sns.clustermap(ordered_arr.T, yticklabels=True, xticklabels=50, col_cluster=False, row_cluster=False,
                               cmap='jet', vmin=-3, vmax=3, cbar_pos=(0.02, 0.795, 0.03, 0.2), figsize=(8, 9))
    else:
        ordered_arr = arr
        sns.set(font_scale=0.8)
        hm = sns.clustermap(ordered_arr.T, yticklabels=True, xticklabels=50, col_cluster=False,
                               cmap='jet', vmin=-3, vmax=3, cbar_pos=(0.02, 0.795, 0.03, 0.2), figsize=(8, 9))
    return ordered_arr, hm


### all neurons heatmap yticklabel (make yticklabel easier to see)
def re_name_all(arr):
    name_list = arr.columns.tolist()
    for i in range(len(name_list)):
        if i % 3 == 0:
            name_list[i] = name_list[i]  # 第一组：1+3*n
        elif i % 3 == 1:
            name_list[i] = '---------' + name_list[i]   # 第二组：2+3*n
        else:
            name_list[i] = '-------------------' + name_list[i]   # 第三组：3+3*n
    print(name_list)
    re_name = np.array(name_list)
    return re_name


### all neurons corr_hm yticklabel (re_name means with ___)
def re_name_all_corr(co_hm, arr):
    co_ylabels = arr.index[co_hm.dendrogram_row.reordered_ind]
    co_names = arr.iloc[:, co_ylabels].columns.values
    co_names2 = co_names.copy()
    for i in range(len(co_names2)):
        if i % 3 == 0:
            co_names2[i] = co_names2[i]  # 第一组：1+3*n
        elif i % 3 == 1:
            co_names2[i] = '---------' + co_names2[i]   # 第二组：2+3*n
        else:
            co_names2[i] = '-------------------' + co_names2[i]   # 第三组：3+3*n
    print(co_names2)
    re_co_names = np.array(co_names2)
    return co_names, re_co_names


### mean all neurons cor_heatmap yticklabel (make yticklabel easier to see)
def re_name_mean(name_list):  ## 一维
    for i in range(len(name_list)):
        if i % 3 == 0:
            name_list[i] = name_list[i]  # 第一组：1+3*n
        elif i % 3 == 1:
            name_list[i] = '---------' + name_list[i]   # 第二组：2+3*n
        else:
            name_list[i] = '-------------------' + name_list[i]   # 第三组：3+3*n
    print(name_list)
    re_name = np.array(name_list)
    return re_name


### 概率分布直方图
def histplot(data, label):
    dt = np.tril(data, k=-1)
    dt2 = dt.flatten()
    dt3 = [n for n in dt2 if n != 0]
    sns.histplot(dt3, stat='probability', label=label, binrange=(-1, 1), binwidth=0.02)

######### mean for target and all neurons in young and old worms ######
### young, target:
def young_mean_tar(tar):
    tarN = tar.iloc[:, 1]
    tarNames = np.array(tarN)
    Mt = len(tarNames)
    samples = list(range(1, 9))
    nsamples = len(samples)

    coarray = np.zeros((nsamples, Mt, Mt))
    coarray[:, :, :] = np.nan  ##  三维数组
    folder_path = 'D:/Connectivity/metadata'

    for samplei in range(nsamples):
        file = 'tar_cor_sample' + str(samplei) + '.csv'
        name = 'tar_name_sample' + str(samplei) + '.csv'
        file_path = os.path.join(folder_path, file)
        name_path = os.path.join(folder_path, name)
        with open(file_path, 'r') as cor:
            data = pd.read_csv(cor, index_col=0, header=None, skiprows=1)  ##  dataframe
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
    return coarray, tarNames


### old, target:
def old_mean_tar(old_tar):
    old_tarN = old_tar.iloc[:, 1]
    old_tarNames = np.array(old_tarN)
    old_Mt = len(old_tarNames)
    old_samples = list(range(1, 8))
    old_nsamples = len(old_samples)

    old_coarray = np.zeros((old_nsamples, old_Mt, old_Mt))
    old_coarray[:, :, :] = np.nan  ##  三维数组
    folder_path = 'D:/Connectivity/metadata'

    for old_samplei in range(old_nsamples):
        old_file = 'old_tar_cor_sample' + str(old_samplei) + '.csv'
        old_name = 'old_tar_name_sample' + str(old_samplei) + '.csv'
        old_file_path = os.path.join(folder_path, old_file)
        old_name_path = os.path.join(folder_path, old_name)
        with open(old_file_path, 'r') as old_cor:
            old_data = pd.read_csv(old_cor, index_col=0, header=None, skiprows=1)  ##  dataframe
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
    return old_coarray, old_tarNames


### young, all:
def young_mean(all):
    allN = all.iloc[:, 1]
    allNames = np.array(allN)
    Ma = len(allNames)
    allsamples = list(range(1, 9))
    n_allsamples = len(allsamples)

    allcoarray = np.zeros((n_allsamples, Ma, Ma))
    allcoarray[:, :, :] = np.nan  ##  三维数组
    folder_path = 'D:/Connectivity/metadata'

    for allsamplei in range(n_allsamples):
        all_file = 'cor_sample' + str(allsamplei) + '.csv'
        all_name = 'name_sample' + str(allsamplei) + '.csv'
        all_file_path = os.path.join(folder_path, all_file)
        all_name_path = os.path.join(folder_path, all_name)
        with open(all_file_path, 'r') as all_cor:
            all_data = pd.read_csv(all_cor, index_col=0, header=None, skiprows=1)  ##  dataframe
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
    return allcoarray, allNames

### old, all:
def old_mean(old_all):
    old_allN = old_all.iloc[:, 1]
    old_allNames = np.array(old_allN)
    old_Ma = len(old_allNames)
    old_allsamples = list(range(1, 8))
    old_n_allsamples = len(old_allsamples)

    old_allcoarray = np.zeros((old_n_allsamples, old_Ma, old_Ma))
    old_allcoarray[:, :, :] = np.nan  ##  三维数组
    folder_path = 'D:/Connectivity/metadata'

    for old_allsamplei in range(old_n_allsamples):
        old_all_file = 'old_cor_sample' + str(old_allsamplei) + '.csv'
        old_all_name = 'old_name_sample' + str(old_allsamplei) + '.csv'
        old_all_file_path = os.path.join(folder_path, old_all_file)
        old_all_name_path = os.path.join(folder_path, old_all_name)
        with open(old_all_file_path, 'r') as old_all_cor:
            old_all_data = pd.read_csv(old_all_cor, index_col=0, header=None, skiprows=1)  ##  dataframe
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
    return old_allcoarray, old_allNames

def hm_tar(coarray, tarNames):
    used_tar_cells = np.where(~np.isnan(coarray).all(axis=(0, 1)))[0]
    used_tar_names = tarNames[used_tar_cells]
    used_tar_coarray = coarray[:, used_tar_cells[:, None], used_tar_cells[None, :]]
    cor_tar_mean = np.nanmean(used_tar_coarray, axis=0).squeeze()
    ### 沿第一纬度计算平均值（即样本间），squeeze删除第一维度

    ## correlation map
    hm_cor_tar_mean = sns.clustermap(cor_tar_mean, xticklabels=used_tar_names, yticklabels=used_tar_names,
                                     col_cluster=False, row_cluster=False, cmap='jet', vmin=-1, vmax=1,
                                     cbar_pos=(0.1, 0.595, 0.03, 0.2))
    return hm_cor_tar_mean, cor_tar_mean


def hm_all(allcoarray, allNames):
    used_cells = np.where(~np.isnan(allcoarray).all(axis=(0, 1)))[0]
    used_names = allNames[used_cells]
    used_coarray = allcoarray[:, used_cells[:, None], used_cells[None, :]]
    cor_mean = np.nanmean(used_coarray, axis=0).squeeze()
    ### 在所有sample中只出现一次的neuron也包括，但可能不存在相应的cor,即cor_mean中会出现NaN

    temp_cor_mean = np.nan_to_num(cor_mean)  ## 处理nan,暂变为0，以便聚类
    Y = pdist(temp_cor_mean)
    Z = linkage(Y)
    Z_tree = optimal_leaf_ordering(Z, Y)
    Z_index = leaves_list(Z_tree)
    cor_mean_ordered = cor_mean[Z_index][:, Z_index]
    used_names_ordered = used_names[Z_index]

    ## correlation map
    sns.set(font_scale=0.8)
    hm_cor_mean = sns.clustermap(cor_mean_ordered, xticklabels=True, yticklabels=True, col_cluster=False,
                                 row_cluster=False, cmap='jet', vmin=-1, vmax=1, cbar_pos=(0.02, 0.79, 0.03, 0.2),
                                 figsize=(6.7, 7))
    return hm_cor_mean, used_names_ordered




### statistic diff: U-tset
def ranksum(A, B):
    n = A.shape[1]
    statistic = np.empty((n,))
    p_value = np.empty((n,))
    for i in range(A.shape[1]):
        va = A[:, i][~np.isnan(A[:, i])]
        vb = B[:, i][~np.isnan(B[:, i])]
        stat, p = ranksums(va, vb, 'two-sided')
        statistic[i] = stat
        p_value[i] = p
    return p_value, statistic


def p2a(p_value):
    aster = []
    for i in range(len(p_value)):
        if p_value[i] < 0.0001:
            aster.append('****')
        elif p_value[i] < 0.001:
            aster.append('***')
        elif p_value[i] < 0.01:
            aster.append('**')
        else:
            aster.append('*')
    return aster
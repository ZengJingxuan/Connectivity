import pandas as pd
import numpy as np
import matplotlib as mpl
from scipy.signal import medfilt
mpl.use('TkAgg')
from scipy.optimize import curve_fit


def de_noise(mat):
    mat = mat
    new = []
    for i in mat:
        new.append(medfilt(i, 5))
    return np.asarray(new)

def rm_zero(x, y):
    dt = list(zip(x, y))
    dt = [n for n in dt if n[1] != 0]
    x, y = zip(*dt)
    return np.asarray(x), np.asarray(y)

def de_trend(mat):
    def fun(x, a, b, c):
        return c * np.exp(-a * x) + b
    time_series = np.linspace(0, mat.shape[1] - 1, num=mat.shape[1])
    new = []
    for data in mat:
        # remove zero point
        x, y = rm_zero(list(time_series), list(data))
        f, err = curve_fit(fun, x, y, p0=[0.003, 0.5, 0.5], maxfev=5000)
        curve = fun(time_series, f[0], f[1], f[2])
        new.append(curve)
    new = np.asarray(new)
    mat = mat / new
    return mat, new

def norm(arr):
    mean = np.mean(arr,axis=0)
    std = np.std(arr,axis = 0)
    return (arr - mean) / std


def main(ch1_p,ch2_p):
    ch1 = pd.read_csv(ch1_p, index_col=0)
    ch2 = pd.read_csv(ch2_p, index_col=0)
    name = ch1.columns
    ch1_d = ch1.values
    ch2_d = ch2.values
    fret_hm = ch2_d / ch1_d
    fret_hm = fret_hm.T
    fret_hm_dn = de_noise(fret_hm)
    fret_hm_dt, curve = de_trend(fret_hm_dn)
    fret_hm_nm = norm(fret_hm_dt.T)
    fret = pd.DataFrame(fret_hm_nm, columns=name)
    return fret


### outliers detect and replace    after y1 = zpr.main(ch1_p, ch2_p)  in main
def rep_outl(arr):
    for col in arr.columns[1:]:
        mean = np.mean(arr[col])
        std = np.std(arr[col])
        threshold = mean + 2 * std
        outlier_indices = np.where(arr[col] > threshold)[0]
        for index in outlier_indices:
            left = max(index - 2, 0)  # 左侧取2个点
            right = min(index + 2, len(arr) - 1)  # 右侧取2个点
            neighbors = arr[col][left:right + 1]
            median = np.median(neighbors)
            arr[col][index] = median
    return arr


#### select target neurons
def select_tar(ch1_p,arr):
    ch1 = pd.read_csv(ch1_p, index_col=0)
    name = ch1.columns
    targets = np.array(['AVAL','AVAR','AVEL','AVER','RIML','RIMR','AIBL','AIBR','RIBL','RIBR','RMEL','RMER','AVBL','AVBR'])
    matches = [];
    for query in targets:
      print(query)
      match = np.argwhere(name==query)
      if len(match) > 0:
       print(match)
       matches.append(match[0][0])
    print(matches)
    arr = arr.iloc[:, matches]
    return arr

### all neurons heatmap yticklabel
def re_order_all(ch1_p,hm):
    hm_ylabel = hm.dendrogram_row.reordered_ind
    ch1 = pd.read_csv(ch1_p, index_col=0)
    re_ch1 = ch1.iloc[:, hm_ylabel]
    name_list = re_ch1.columns.tolist()
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


### 概率分布直方图，第一行转为一维数组，第二行去除重复值（三角对称），第三行 n!=0(n不为0的被保留）
def histplot(data,label):
    dt = data.flatten()
    dt = set(dt)
    dt = [n for n in dt if n != 0]
    sns.histplot(dt,stat='probability',label=label)






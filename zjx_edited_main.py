import pandas as pd
import zjx_edited_preprocess as zpr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from scipy.stats import kstest
from scipy.signal import savgol_filter
from collections import Counter
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests



ch1_p = r"230421-9DA-W1-ch1.csv"
ch2_p = r"230421-9DA-W1-ch2.csv"
y1 = zpr.main(ch1_p, ch2_p)

# ch1_p = r"230102-W3-ch1.csv"
# ch2_p = r"230102-W3-ch2.csv"
# # generate normalized mat
# y1 = zpr.main(ch1_p, ch2_p)

# select target neurons
y1_tar = zpr.select_tar(ch1_p, ch2_p, y1)
tar_name_y1 = y1_tar.columns.values
### 将字体设为1.5倍
sns.set(font_scale=1.5)
### heatmap
hm_y1_tar = sns.clustermap(y1_tar.T, yticklabels=tar_name_y1, xticklabels=50, col_cluster=False, row_cluster=False, cmap='jet',vmin=-3,vmax=3,cbar_pos=(0.13, 0.595, 0.03, 0.2))
### correlation:
co_y1_tar = np.corrcoef(y1_tar.T)
co_hm_y1_tar = sns.clustermap(co_y1_tar, xticklabels=tar_name_y1, yticklabels=tar_name_y1, col_cluster=False, row_cluster=False, cmap='jet',vmin=-1, vmax=1,cbar_pos=(0.1, 0.595, 0.03, 0.2))
## values for correlation map


pd.DataFrame(co_y1_tar).to_csv("old_tar_cor_sample3.csv")
pd.DataFrame(tar_name_y1).to_csv("old_tar_name_sample3.csv")


### all neurons
sns.set(font_scale=0.8)
ordered_y1, hm_y1 = zpr.cluster(y1)     ### choose True & False (if use optimal_leaf_ordering)
### yticklabel easier to see
re_name_ordered_y1 = zpr.re_name_all(ordered_y1)
hm_y1.ax_heatmap.set_yticklabels(re_name_ordered_y1)

# heatmap of co_eff matrix
co_eff_y1 = np.corrcoef(ordered_y1.T)  # co_eff matrix hm聚类方式不改变coeff结果
co_y1 = sns.clustermap(co_eff_y1, xticklabels=True, yticklabels=True, cmap='jet', vmin=-1, vmax=1, cbar_pos=(0.02, 0.79, 0.03, 0.2), figsize=(6.7, 8))
co_y1_xnames, co_y1_ynames = zpr.re_name_all_corr(co_y1, ordered_y1)
co_y1.ax_heatmap.set_yticklabels(co_y1_ynames)
co_y1.ax_heatmap.set_xticklabels(co_y1_xnames)
## values for correlation map

zpr.histplot(co_eff_y1, label='0421-9DA-W1')
plt.ylabel('Probability', fontsize=18)
plt.legend(fontsize=18)
plt.show()


# only save annotated neurons
y1_all_names = y1.columns.values.tolist()
y1_anno_names = [name for name in y1_all_names if re.match(r'^[A-Za-z]', name)]
y1_anno = y1[y1_anno_names]
name_anno_y1 = y1_anno.columns.values
co_anno_y1 = np.corrcoef(y1_anno.T)
pd.DataFrame(name_anno_y1).to_csv("old_name_sample3.csv")
pd.DataFrame(co_anno_y1).to_csv("old_cor_sample3.csv")     ##记得old


#######  target neurons, smooth, count times, t-test, multipletest ######
y2 = savgol_filter(y1_tar, 61, 3, axis=0)
y3 = y2[100:900, :]
thresh, times, cross = zpr.comp2thr(y3)
pd.DataFrame(times).to_csv("old_times_sample2.csv")



tar = pd.read_csv(r"D:/Connectivity/metadata_states/target_neurons.csv")
times = zpr.young_times(tar)
old_times = zpr.old_times(tar)

sta_t, p_t = zpr.ttest(times, old_times)
# reject, adjusted_p, _, _ = multipletests(p_t, 0.05, method='fdr_bh')
used_times = np.nanmean(times, axis=0).squeeze()
old_used_times = np.nanmean(old_times, axis=0).squeeze()
ind_t = np.where((p_t < 0.05) & (p_t != 0))
times2 = used_times[ind_t]
old_times2 = old_used_times[ind_t]
p_t2 = p_t[ind_t]
aster_t = zpr.p2a(p_t2)

N = tar.values
NN = N[:, 1]
Name = NN.T[ind_t[0]]
###
X = np.arange(len(used_times))
width = 0.25
plt.figure(figsize=(8, 6))
rect_Y = plt.bar(X - width/2, used_times, width, label='young')
rect_O = plt.bar(X + width/2, old_used_times, width, label='old')

plt.grid(axis='y', alpha=0.3)
plt.ylabel("times", fontsize=18)
plt.xticks(X, NN, rotation=45, fontsize=18)
plt.legend(fontsize=14)
plt.show()





################   mean for neural pairs correlation arcoss samples  ###################

## young, tar
tar = pd.read_csv(r"D:/Connectivity/metadata/targetneurons.csv")
coarray, tarNames = zpr.young_mean_tar(tar)
hm_young_tar, cor_tar = zpr.hm_tar(coarray, tarNames)

## old, tar
old_tar = pd.read_csv(r"D:/Connectivity/metadata/targetneurons.csv")
old_coarray, old_tarNames = zpr.old_mean_tar(old_tar)
hm_old_cor_tar, old_cor_tar = zpr.hm_tar(old_coarray, old_tarNames)


zpr.histplot(cor_tar, label='1DA-target')
zpr.histplot(old_cor_tar, label='9DA-target')
plt.ylabel('Probability', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=15)
plt.show()

pd.DataFrame(old_cor_tar).to_csv("9DA-mean_tar_cor.csv")
pd.DataFrame(cor_tar).to_csv("1DA-mean_tar_cor.csv")


## young, all
all = pd.read_csv(r"D:/Connectivity/metadata/allneurons.csv")
allcoarray, allNames = zpr.young_mean(all)         ### 3d all (for p_value)
cor = np.nanmean(allcoarray, axis=0).squeeze()     ### 2d all (for save csv, diff and histplot)

hm_young_all, used_names_ordered = zpr.hm_all(allcoarray, allNames)
used_xnames = used_names_ordered.copy()
used_ynames = zpr.re_name_mean(used_names_ordered)
hm_young_all.ax_heatmap.set_yticklabels(used_ynames, fontsize=7)
hm_young_all.ax_heatmap.set_xticklabels(used_xnames, fontsize=7)

## old, all
old_all = pd.read_csv(r"D:/Connectivity/metadata/allneurons.csv")
old_allcoarray, old_allNames = zpr.old_mean(old_all)     ### 3d all (for p_value)
old_cor = np.nanmean(old_allcoarray, axis=0).squeeze()   ### 2d all (for save csv, diff and histplot)

hm_old_all, old_used_names_ordered = zpr.hm_all(old_allcoarray, old_allNames)
old_used_xnames = old_used_names_ordered.copy()
old_used_ynames = zpr.re_name_mean(old_used_names_ordered)
hm_old_all.ax_heatmap.set_yticklabels(old_used_ynames, fontsize=7)
hm_old_all.ax_heatmap.set_xticklabels(old_used_xnames, fontsize=7)


zpr.histplot(cor, label='1DA')
zpr.histplot(old_cor, label='9DA')
plt.ylabel('Probability', fontsize=22)
plt.grid(axis='y', alpha=0.3)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=22)
plt.show()


#############  proportion of neg and pos  ######################################################
allsamples = list(range(1, 9))
old_allsamples = list(range(1, 7))
ypneg, yppos = zpr.young_pro(allsamples)
opneg, oppos = zpr.old_pro(old_allsamples)
y_pos_mean = np.mean(yppos)
y_neg_mean = np.mean(ypneg)
o_pos_mean = np.mean(oppos)
o_neg_mean = np.mean(opneg)
psta, pp = ttest_ind(yppos, oppos)
nsta, np = ttest_ind(ypneg, opneg)
neg_reject, neg_adj_p, _, _ = multipletests(np, 0.05, method='fdr_bh')
pos_reject, pos_adj_p, _, _ = multipletests(pp, 0.05, method='fdr_bh')

x = 1
width = 0.25
plt.figure(figsize=(8, 6))
rect_y = plt.bar(x - width/2, y_neg_mean, width, label='young')
rect_o = plt.bar(x + width/2, o_neg_mean, width, label='old')
plt.grid(axis='y', alpha=0.3)
plt.ylabel("proportion", fontsize=18)
plt.xticks([])
plt.legend()
plt.show()



###########   k-s test for correlation distribution, cor: 2D    ###################
### for all
cor1 = np.tril(cor)
cor2 = [n for n in cor1.flatten() if n != 0]
cor3 = np.array(cor2)
old_cor1 = np.tril(old_cor)
old_cor2 = [n for n in old_cor1.flatten() if n != 0]
old_cor3 = np.array(old_cor2)
d, p4d = kstest(cor3, old_cor3, 'two-sided')
##  too many varibles p,   p4d: p value for distance

### for target
# cor_tar1 = np.tril(cor_tar)
# cor_tar2 = [n for n in cor_tar1.flatten() if n != 0]
# cor_tar3 = np.array(cor_tar2)
# old_cor_tar1 = np.tril(old_cor_tar)
# old_cor_tar2 = [n for n in old_cor_tar1.flatten() if n != 0]
# old_cor_tar3 = np.array(old_cor_tar2)
# d, p4d = kstest(cor_tar3, old_cor_tar3, 'two-sided')
##  too many varibles p,   p4d: p value for distance





### U-test,  allcoarray: 3D     young: 8, old: 7;    for n:  target:16;  all: 232  ##############
Young = allcoarray.reshape((8, -1))
Old = old_allcoarray.reshape((6, -1))
sta2, p2 = zpr.ranksum(Young, Old)
p3 = p2.reshape(232, 232)
sta3 = sta2.reshape(232, 232)
p = np.tril(p3, k=-1)
sta = np.tril(sta3, k=-1)

# p5 = p.flatten()
# p6 = [n for n in p5 if n != 0]
# p7 = np.array(p6)
# reject, adjusted_p, _, _ = multipletests(p7, 0.05, method='fdr_bh')

pd.DataFrame(p).to_csv("P_all.csv")



############# visualization ###########################
young = r"D:/Connectivity/data-new/1DA-mean_tar_cor.csv"
old = r"D:/Connectivity/data-new/9DA-mean_tar_cor.csv"
names = r"D:/Connectivity/data-new/tar_names.csv"
# young = r"D:/Connectivity/data-new/1DA-mean_cor.csv"
# old = r"D:/Connectivity/data-new/9DA-mean_cor.csv"


y = pd.read_csv(young, index_col=0)
o = pd.read_csv(old, index_col=0)
n = pd.read_csv(names, index_col=0)        ## dataframe
ov = o.values                              ## array
yv = y.values
nv = n.values
##########   p-value heatmap,   only p<0.05 show colors  ################
cmap = plt.cm.hot
threshold = 0.05
pnon = ~np.isnan(p3).all(axis=1)
p5 = p3[pnon]
pn = ~np.isnan(p3).all(axis=0)
p6 = p5[:, pn]
new_p6 = np.where(p6 > threshold, threshold, p6)
plt.imshow(new_p6, cmap=cmap, vmin=0, vmax=threshold, interpolation='nearest')
plt.colorbar()  # 显示颜色条
plt.grid(alpha=0.3)
plt.yticks((np.arange(len(tarNames))), tarNames, fontsize=12)
plt.xticks((np.arange(len(tarNames))), tarNames, rotation=45, fontsize=12)
plt.show()
#############################################################################
ind = np.where((p < 0.05) & (p != 0))
# ind = np.where((p < 0.05) & (p != 0) & (p > 0.01))
y2 = yv[ind]
o2 = ov[ind]
p4 = p[ind]
aster = zpr.p2a(p4)

ind2 = np.where((p < 0.01) & (p != 0))
y3 = yv[ind2]
o3 = ov[ind2]
p4 = p[ind2]
aster2 = zpr.p2a(p4)

##  ind2 if choose **
row = n.T[ind[0]]
col = n.T[ind[1]]
row2 = row.values
col2 = col.values
col3 = col2.copy()
for i in range(len(col2)):
    col3[i] = '-' + col2[i]
print(col3)

name = row2.copy()
for i in range(len(row2)):
    name[i] = row2[i] + col3[i]
name2 = name.tolist()
name3 = ','.join(str(i) for i in name2)
name4 = name3.split(",")

##### number of changed neurons
chan = np.concatenate((row2, col2))
counter = Counter(chan)  # 使用 Counter 统计元素出现次数
for element, count in counter.items():
    print(element, count)
count = np.array(counter.most_common())



### for all **, variable names change:  y2--y3; o2--o3
x = np.arange(len(y2))
width = 0.25
plt.figure(figsize=(8, 6))
rect_y = plt.bar(x - width/2, y2, width, label='young')
rect_o = plt.bar(x + width/2, o2, width, label='old')

plt.grid(axis='y', alpha=0.3)
plt.ylabel("correlation value", fontsize=18)
plt.xticks(x, name4, rotation=45, fontsize=11)
# plt.xticks(x, name4, rotation=90, fontsize=7)
plt.legend(fontsize=14)

for xi, oi, val in zip(x, o2, aster):
    plt.text(xi, oi, str(val), ha='right', va='bottom', fontsize=14)
plt.show()
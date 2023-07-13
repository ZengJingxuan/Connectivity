import pandas as pd
import zjx_edited_preprocess as zpr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

# ch1_p = r"230512-9DA-W1-ch1.csv"
# ch2_p = r"230512-9DA-W1-ch2.csv"

ch1_p = r"230102-W3-ch1.csv"
ch2_p = r"230102-W3-ch2.csv"
# generate normalized mat
y1 = zpr.main(ch1_p, ch2_p)

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

pd.DataFrame(co_y1_tar).to_csv("tar_cor_sample3.csv")
pd.DataFrame(tar_name_y1).to_csv("tar_name_sample3.csv")


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

# only save annotated neurons
y1_all_names = ordered_y1.columns.values.tolist()
y1_anno_names = [name for name in y1_all_names if re.match(r'^[A-Za-z]', name)]
y1_anno = ordered_y1[y1_anno_names]
name_anno_y1 = y1_anno.columns.values
co_anno_y1 = np.corrcoef(y1_anno.T)
pd.DataFrame(name_anno_y1).to_csv("name_sample3.csv")
pd.DataFrame(co_anno_y1).to_csv("cor_sample3.csv")     ##记得old


# # select most dynamic neurons
# y1_dy = zpr.select(y1_anno, 60)        ## n = ?
# dy_y1, dy_hm_y1 = zpr.cluster(y1_dy)   ## 可选False
# co_eff_dy_y1 = np.corrcoef(y1_dy.T)
# co_dy_y1 = sns.clustermap(co_eff_dy_y1, xticklabels=True, yticklabels=True, cmap='jet', vmin=-1, vmax=1, figsize=(10, 6.5))
# ### labels
# co_dy_ylabels = y1_dy.index[co_dy_y1.dendrogram_row.reordered_ind]
# co_dy_y1_names = y1_dy.iloc[:, co_dy_ylabels].columns.values
# co_dy_y1.ax_heatmap.set_yticklabels(co_dy_y1_names)
# co_dy_y1.ax_heatmap.set_xticklabels(co_dy_y1_names)
# pd.DataFrame(co_dy_y1_names).to_csv("dy_name_sample6.csv")
# pd.DataFrame(co_eff_dy_y1).to_csv("dy_cor_sample6.csv")


zpr.histplot(co_eff_y1, label='0102-1DA-W3')
plt.legend()
plt.show()



## young, tar
tar = pd.read_csv(r"D:/Connectivity/metadata/targetneurons.csv")
coarray, tarNames = zpr.young_mean_tar(tar)
hm_young_tar = zpr.hm_tar(coarray, tarNames)

## old, tar
old_tar = pd.read_csv(r"D:/Connectivity/metadata/targetneurons.csv")
old_coarray, old_tarNames = zpr.old_mean_tar(old_tar)
hm_old_cor_tar = zpr.hm_tar(old_coarray, old_tarNames)

pd.DataFrame(old_coarray).to_csv("9DA-mean_old_tar_cor.csv")
pd.DataFrame(coarray).to_csv("1DA-mean_tar_cor.csv")


## young, all
all = pd.read_csv(r"D:/Connectivity/metadata/allneurons.csv")
allcoarray, allNames = zpr.young_mean(all)         ### 3d all (for p_value)
cor = np.nanmean(allcoarray, axis=0).squeeze()     ### 2d all (for csv and diff)

hm_young_all, used_names_ordered = zpr.hm_all(allcoarray, allNames)
used_xnames = used_names_ordered.copy()
used_ynames = zpr.re_name_mean(used_names_ordered)
hm_young_all.ax_heatmap.set_yticklabels(used_ynames)
hm_young_all.ax_heatmap.set_xticklabels(used_xnames)

## old, all
old_all = pd.read_csv(r"D:/Connectivity/metadata/allneurons.csv")
old_allcoarray, old_allNames = zpr.old_mean(old_all)     ### 3d all (for p_value)
old_cor = np.nanmean(old_allcoarray, axis=0).squeeze()   ### 2d all (for csv and diff)

hm_old_all, old_used_names_ordered = zpr.hm_all(old_allcoarray, old_allNames)
old_used_xnames = old_used_names_ordered.copy()
old_used_ynames = zpr.re_name_mean(old_used_names_ordered)
hm_old_all.ax_heatmap.set_yticklabels(old_used_ynames)
hm_old_all.ax_heatmap.set_xticklabels(old_used_xnames)



### U-test: young: 8, old: 7;    for n:  target:16;  all: 232
Young = allcoarray.reshape((8, -1))
Old = old_allcoarray.reshape((7, -1))
p2, sta2 = zpr.ranksum(Young, Old)
p3 = p2.reshape(232, 232)
sta3 = sta2.reshape(232, 232)
p = np.tril(p3, k=-1)
sta = np.tril(sta3, k=-1)

pd.DataFrame(p).to_csv("P_tar.csv")



############# visualization ###########################
young = r"1DA-mean_tar_cor.csv"
old = r"9DA-mean_tar_cor.csv"
names = r"1DA-mean_tar_names.csv"
# young = r"1DA_cor.csv"
# old = r"9DA_cor.csv"
# names = r"1DA_names.csv"

y = pd.read_csv(young, index_col=0)
o = pd.read_csv(old, index_col=0)
n = pd.read_csv(names, index_col=0)        ## dataframe
ov = o.values                              ## array
yv = y.values
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


row = n.T[ind[0]]
col = n.T[ind[1]]
row2 = row.values
col2 = col.values
col3 = col2.copy()
for i in range(len(col2)):
    col3[i] = '—' + col2[i]
print(col3)

name = row2.copy()
for i in range(len(row2)):
    name[i] = row2[i] + col3[i]
name2 = name.tolist()
name3 = ','.join(str(i) for i in name2)
name4 = name3.split(",")

# pd.DataFrame(name).to_csv("p_one-aster_name.csv")


### for all and **, y2--y3; o2--o3
x = np.arange(len(y2))
width = 0.25
plt.figure(figsize=(8, 6))
rect_y = plt.bar(x - width/2, y2, width, label='young')
rect_o = plt.bar(x + width/2, o2, width, label='old')

plt.grid(axis='y', alpha=0.3)
plt.ylabel("value", fontsize=18)
plt.xticks(x, name4, rotation=45, fontsize=11)
# plt.xticks(x, name4, rotation=90, fontsize=7)
plt.legend(fontsize=14)

for xi, oi, val in zip(x, o2, aster):
    plt.text(xi, oi, str(val), ha='right', va='bottom', fontsize=14)
plt.show()
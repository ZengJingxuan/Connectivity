import pandas as pd
import matplotlib as mpl
import zjx_edited_preprocess as zpr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

ch1_p = r"230102-W3-ch1.csv"
ch2_p = r"230102-W3-ch2.csv"    #  "230421-9DA-W1-ch2.csv"

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


zpr.histplot(co_eff_y1, label='0102-1DA-W3')
plt.legend()
plt.show()

import pandas as pd
import matplotlib as mpl
import zjx_edited_preprocess as zpr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


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
hm_y1_tar = sns.clustermap(y1_tar.T, yticklabels=tar_name_y1, xticklabels=50, col_cluster=False, row_cluster=False,cmap='jet',vmin=-3,vmax=3,cbar_pos=(0.13, 0.595, 0.03, 0.2))
### correlation:
co_y1_tar = np.corrcoef(y1_tar.T)
co_hm_y1_tar = sns.clustermap(co_y1_tar, xticklabels=tar_name_y1, yticklabels=tar_name_y1, col_cluster=False, row_cluster=False, cmap='jet',vmin=-1, vmax=1,cbar_pos=(0.1, 0.595, 0.03, 0.2))
## values for correlation map
v_y1_tar = co_hm_y1_tar.data2d.values


### all neurons
sns.set(font_scale=0.8)
ordered_y1 = zpr.cluster(y1)
hm_y1 = sns.clustermap(y1.T, yticklabels=True, xticklabels=50, col_cluster=False, row_cluster=False, cmap='jet',vmin=-3, vmax=3,cbar_pos=(0.02, 0.795, 0.03, 0.2))
### yticklabel easier to see
re_name_ordered_y1 = zpr.re_name_all(ordered_y1)
hm_y1.ax_heatmap.set_yticklabels(re_name_ordered_y1)
plt.show()

co_eff_y1 = np.corrcoef(y1.T)  # co_eff matrix
co_y1 = sns.clustermap(co_eff_y1, cmap='jet', vmin=-1, vmax=1, cbar_pos=(0.02, 0.79, 0.03, 0.2))
# heatmap of co_eff matrix


zpr.histplot(co_eff_y1, label='0102-1DA-W3')
plt.legend()
plt.show()

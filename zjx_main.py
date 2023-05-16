import pandas as pd
import matplotlib as mpl
import zjx_preprocess as zpr
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import connectome as connect
import networkx as nx


ch1_p = r"230102-W3-ch1.csv"
ch2_p = r"230102-W3-ch2.csv"
ch1 = pd.read_csv(ch1_p, index_col=0)
name = ch1.columns
y1 = zpr.main(ch1_p, ch2_p)  # generate normalized mat
y1 = zpr.rep_outl(y1)       # replace outliers
y1_tar = zpr.select_tar(ch1_p,y1)
tar_name = y1_tar.columns.values

### heatmap
sns.clustermap(y1_tar.T, yticklabels=tar_name, col_cluster=False, row_cluster=False,cmap='jet',vmin=-3,vmax=3)
### correlation:
co_y1_tar = np.corrcoef(y1_tar.T)
sns.clustermap(co_y1_tar, xticklabels=tar_name, yticklabels=tar_name, col_cluster=False, row_cluster=False, cmap='jet',vmin=-1, vmax=1)
##

### all neurons
sns.clustermap(y1.T, yticklabels=True, col_cluster=False, cmap='jet',vmin=-3, vmax=3)
co_eff = np.corrcoef(y1.T)  # co_eff matrix
sns.clustermap(co_eff, cmap='jet',vmin=-1, vmax=1)  # heatmap of co_eff matrix
pr.histplot(co_eff, label='0102-1DA-W3')
plt.legend()
plt.show()
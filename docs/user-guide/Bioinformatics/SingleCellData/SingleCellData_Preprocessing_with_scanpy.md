# Installation

# Read and preprocessing the datasets

The dataset contains control and IFN-beta stimulated cells. We use this as the query dataset.


```python
from platform import python_version

print(python_version())
```

    3.11.7



```python
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import decoupler as dc
import session_info
```


```python
import matplotlib.pyplot as plt

sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
#sc.logging.print_versions()

sc.settings.set_figure_params(dpi=80)
%matplotlib inline
```


```python
sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')
```

    /Users/lamine/anaconda3/envs/env/lib/python3.11/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm


    scanpy==1.9.6 anndata==0.10.4 umap==0.5.5 numpy==1.26.3 scipy==1.12.0 pandas==2.2.0 scikit-learn==1.4.0 statsmodels==0.14.1 pynndescent==0.5.11



```python
results_file = '/path/Data/file_merged.h5ad'  
#the file that will store the analysis results
```


```python
SS001 = sc.read_10x_mtx(
    '/path/SS001/outs/filtered_gene_bc_matrices/GRCh38',  # the directory with the `.mtx` file
    var_names='gene_symbols',                # use gene symbols for the variable names (variables-axis index)
    cache=True)
SS002 = sc.read_10x_mtx(
    '/path/SS002/outs/filtered_gene_bc_matrices/GRCh38',  # the directory with the `.mtx` file
    var_names='gene_symbols',                # use gene symbols for the variable names (variables-axis index)
    cache=True)  
SS003 = sc.read_10x_mtx(
    '/path/SS003/outs/filtered_gene_bc_matrices/GRCh38',  # the directory with the `.mtx` file
    var_names='gene_symbols',                # use gene symbols for the variable names (variables-axis index)
    cache=True)  
SS004 = sc.read_10x_mtx(
    '/path/SS004/outs/filtered_gene_bc_matrices/GRCh38',  # the directory with the `.mtx` file
    var_names='gene_symbols',                # use gene symbols for the variable names (variables-axis index)
    cache=True) 

SS005 = sc.read_10x_mtx(
    '/path/SS005/outs/filtered_gene_bc_matrices/GRCh38',  # the directory with the `.mtx` file
    var_names='gene_symbols',                # use gene symbols for the variable names (variables-axis index)
    cache=True)
SS006 = sc.read_10x_mtx(
    '/path/SS006/outs/filtered_gene_bc_matrices/GRCh38',  # the directory with the `.mtx` file
    var_names='gene_symbols',                # use gene symbols for the variable names (variables-axis index)
    cache=True)  
SS007 = sc.read_10x_mtx(
    '/path/SS007/outs/filtered_gene_bc_matrices/GRCh38',  # the directory with the `.mtx` file
    var_names='gene_symbols',                # use gene symbols for the variable names (variables-axis index)
    cache=True)  
SS008 = sc.read_10x_mtx(
    '/path/SS008/outs/filtered_gene_bc_matrices/GRCh38',  # the directory with the `.mtx` file
    var_names='gene_symbols',                # use gene symbols for the variable names (variables-axis index)
    cache=True)  

```


```python
file_merged = ad.concat([SS001, SS002, SS003, SS004, SS005, SS006, SS007, SS008], label="batch", 
                            keys=["SS001", "SS002", "SS003", "SS004", "SS005", "SS006", "SS007", "SS008"])
file_merged
```


```python
file_merged.obs['condition'] = 'WT'
file_merged.obs
```


```python
Trait = file_merged.obs.batch=='D6'
file_merged.obs.loc[Trait, 'condition'] = 'IRF3&IFNb&vRNA'
file_merged.obs['condition']
```


```python
file_merged.var_names_make_unique()  # this is unnecessary if using `var_names='gene_ids'` in `sc.read_10x_mtx`
file_merged
```

## Detect variables genes

Variable genes can be detected across the full dataset, but then we run the risk of getting many batch-specific genes that will drive a lot of the variation. Or we can select variable genes from each batch separately to get only celltype variation. In the dimensionality reduction exercise, we already selected variable genes, so they are already stored in file_merged.var.highly_variable


```python
sc.pl.highest_expr_genes(file_merged, n_top=20, )
```


```python
sc.pp.filter_cells(file_merged, min_genes=200)
sc.pp.filter_genes(file_merged, min_cells=3)
```


```python
file_merged.var['mt'] = file_merged.var_names.str.startswith('MT-')  
# annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(file_merged, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
```


```python
file_merged
```


```python
#leg = ax.legend() 
sc.pl.violin(file_merged, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
             jitter=0.4, multi_panel=True)
```


```python
file_merged = file_merged[file_merged.obs.n_genes_by_counts < 8500, :]
file_merged = file_merged[file_merged.obs.pct_counts_mt < 20 , :]
```


```python
file_merged
```


```python
file_merged.layers['counts']= file_merged.X.copy()
sc.pp.normalize_total(file_merged, target_sum=1e4)
```


```python
sc.pp.log1p(file_merged)
```


```python
sc.pp.highly_variable_genes(file_merged, min_mean=0.0125, max_mean=3, min_disp=0.5)
```


```python
var_genes_all = file_merged.var.highly_variable

print("Highly variable genes: %d"%sum(var_genes_all))
```

Detect variable genes in each dataset separately using the batch_key parameter.


```python
sc.pp.highly_variable_genes(file_merged, min_mean=0.0125, max_mean=3, min_disp=0.5, batch_key = 'batch')

print("Highly variable genes intersection: %d"%sum(file_merged.var.highly_variable_intersection))

print("Number of batches where gene is variable:")
print(file_merged.var.highly_variable_nbatches.value_counts())

var_genes_batch = file_merged.var.highly_variable_nbatches > 0
```

Compare overlap of the variable genes.


```python
print("Any batch var genes: %d"%sum(var_genes_batch))
print("All data var genes: %d"%sum(var_genes_all))
print("Overlap: %d"%sum(var_genes_batch & var_genes_all))
print("Variable genes in all batches: %d"%sum(file_merged.var.highly_variable_nbatches == 6))
print("Overlap batch instersection and all: %d"%sum(var_genes_all & file_merged.var.highly_variable_intersection))
```

Select all genes that are variable in at least 2 datasets and use for remaining analysis.


```python
var_select = file_merged.var.highly_variable_nbatches > 2
var_genes = var_select.index[var_select]
len(var_genes)
```


```python
sc.pl.highly_variable_genes(file_merged)
```


```python
file_merged.raw = file_merged
```


```python
file_merged = file_merged[:, file_merged.var.highly_variable]
sc.pp.regress_out(file_merged, ['total_counts', 'pct_counts_mt'])
sc.pp.scale(file_merged, max_value=10)
```

## Data integration : batch effect correction
###  BBKNN
First we will run BBKNN that is implemented in scanpy.


```python
sc.tl.pca(file_merged, svd_solver='arpack')
sc.pl.pca_variance_ratio(file_merged, log=True)
```


```python
file_merged_bbknn = file_merged
```


```python
sc.external.pp.bbknn(file_merged_bbknn, batch_key='batch', n_pcs=30)  

```


```python
sc.pp.neighbors(file_merged, n_neighbors=10, n_pcs=30)
sc.tl.leiden(file_merged)
sc.tl.paga(file_merged)
sc.pl.paga(file_merged, plot=False)  # remove `plot=False` if you want to see the coarse-grained graph
sc.tl.umap(file_merged, init_pos='paga')
sc.tl.tsne(file_merged)
```


```python
sc.pp.neighbors(file_merged_bbknn, n_neighbors=10, n_pcs=30)
sc.tl.leiden(file_merged_bbknn)
sc.tl.paga(file_merged_bbknn)
sc.pl.paga(file_merged_bbknn, plot=False)  # remove `plot=False` if you want to see the coarse-grained graph
sc.tl.umap(file_merged_bbknn, init_pos='paga')
sc.tl.tsne(file_merged_bbknn)
```

We can now plot the un-integrated and the integrated space reduced dimensions.


```python
fig, axs = plt.subplots(2, 2, figsize=(10,8),constrained_layout=True)
sc.pl.tsne(file_merged_bbknn, color="batch", title="BBKNN Corrected tsne", ax=axs[0,0], show=False)
sc.pl.tsne(file_merged, color="batch", title="Uncorrected tsne", ax=axs[0,1], show=False)
sc.pl.umap(file_merged_bbknn, color="batch", title="BBKNN Corrected umap", ax=axs[1,0], show=False)
sc.pl.umap(file_merged, color="batch", title="Uncorrected umap", ax=axs[1,1], show=False)
```


```python
sc.pl.tsne(file_merged, color='batch', title="Uncorrected umap", wspace=.5)
sc.pl.tsne(file_merged_bbknn, color= 'batch', title="BBKNN Corrected umap", wspace=.5)
```


```python
sc.tl.rank_genes_groups(file_merged_bbknn, 'batch', method='t-test')
sc.pl.rank_genes_groups(file_merged_bbknn, n_genes=25, sharey=False)
```


```python
sc.settings.verbosity = 2  # reduce the verbosity
sc.tl.rank_genes_groups(file_merged_bbknn, 'batch', method='wilcoxon')
sc.pl.rank_genes_groups(file_merged_bbknn, n_genes=25, sharey=False)
```


```python
file_merged_bbknn.write(results_file)
```


```python
pd.DataFrame(file_merged_bbknn.uns['rank_genes_groups']['names']).head(10)
```


```python
result = file_merged_bbknn.uns['rank_genes_groups']
groups = result['names'].dtype.names
pd.DataFrame(
    {group + '_' + key[:1]: result[key][group]
    for group in groups for key in ['names', 'pvals']}).head(10)
```


```python
sc.tl.rank_genes_groups(file_merged_bbknn, 'condition', groups=['IRF3&IFNb'], reference='IRF3', method='wilcoxon')
sc.pl.rank_genes_groups(file_merged_bbknn, groups=['IRF3&IFNb'], n_genes=20, fontsize=10)
```


```python
sc.tl.rank_genes_groups(file_merged_bbknn, 'condition', groups=['WT&IFNb'], reference='WT', method='wilcoxon')
sc.pl.rank_genes_groups(file_merged_bbknn, groups=['WT&IFNb'], n_genes=20, fontsize=10)
```


```python
sc.pl.rank_genes_groups_violin(file_merged_bbknn, groups='WT&IFNb', n_genes=8)
```


```python
file_merged_bbknn.obs
```


```python
sc.pl.umap(file_merged_bbknn, color=['condition', 'batch'], wspace=.5)
```


```python
sc.pl.tsne(file_merged_bbknn, color='leiden', legend_loc='on data', title='', frameon=False, save='.pdf')
```


```python
file_merged_bbknn
```


```python
file_merged_bbknn.write(results_file)
```

# Cell type automatic annotation from marker genes

Using ```decoupler```

https://decoupler-py.readthedocs.io/en/latest/notebooks/cell_annotation.html


```python
markers = dc.get_resource('PanglaoDB')
markers
```


```python
# Filter by canonical_marker and human
markers = markers[(markers['human']=='True')&(markers['canonical_marker']=='True')]

# Remove duplicated entries
markers = markers[~markers.duplicated(['cell_type', 'genesymbol'])]
markers
```


```python
dc.run_ora(
    mat=file_merged_bbknn,
    net=markers,
    source='cell_type',
    target='genesymbol',
    min_n=3,
    verbose=True
)
```


```python
file_merged_bbknn
```

Enrichment with Over Representation Analysis (ORA)



```python
file_merged_bbknn.obsm['ora_estimate']
```


```python
acts = dc.get_acts(file_merged_bbknn, obsm_key='ora_estimate')

# We need to remove inf and set them to the maximum value observed for pvals=0
acts_v = acts.X.ravel()
max_e = np.nanmax(acts_v[np.isfinite(acts_v)])
acts.X[~np.isfinite(acts.X)] = max_e

acts
```


```python
sc.pl.umap(acts, color=['NK cells', 'leiden'], cmap='RdBu_r')
sc.pl.violin(acts, keys=['NK cells'], groupby='leiden')
```


```python
df = dc.rank_sources_groups(acts, groupby='leiden', reference='rest', method='t-test_overestim_var')
df
```


```python
n_ctypes = 3
ctypes_dict = df.groupby('group').head(n_ctypes).groupby('group')['names'].apply(lambda x: list(x)).to_dict()
ctypes_dict
```


```python
sc.pl.matrixplot(acts, ctypes_dict, 'leiden', dendrogram=True, standard_scale='var',
                 colorbar_title='Z-scaled scores', cmap='RdBu_r')
```


```python
sc.pl.violin(acts, keys=['T cells', 'B cells', 'Platelets', 'Monocytes', 'NK cells'], groupby='leiden')
```


```python
annotation_dict = df.groupby('group').head(1).set_index('group')['names'].to_dict()
annotation_dict
```


```python
file_merged_bbknn.obs['cell_type'] = [annotation_dict[clust] for clust in file_merged_bbknn.obs['leiden']]
```


```python
# Visualize
sc.pl.umap(file_merged_bbknn, color='cell_type')
```


```python
file_merged_bbknn.obs
```


```python
file_merged_bbknn.write(results_file)
```


```python
session_info.show()
```




<details>
<summary>Click to view session information</summary>
<pre>
-----
anndata             0.10.4
decoupler           1.5.0
matplotlib          3.8.2
numpy               1.26.3
pandas              2.2.0
scanpy              1.9.6
session_info        1.0.0
-----
</pre>
<details>
<summary>Click to view modules imported as dependencies</summary>
<pre>
PIL                         10.2.0
anyio                       NA
appnope                     0.1.3
asttokens                   NA
attr                        23.1.0
attrs                       23.1.0
babel                       2.11.0
brotli                      1.0.9
certifi                     2023.11.17
cffi                        1.16.0
charset_normalizer          2.0.4
comm                        0.2.1
cycler                      0.12.1
cython_runtime              NA
dateutil                    2.8.2
debugpy                     1.6.7
decorator                   5.1.1
defusedxml                  0.7.1
executing                   2.0.1
fastjsonschema              NA
h5py                        3.10.0
idna                        3.4
ipykernel                   6.29.0
jedi                        0.19.1
jinja2                      3.1.2
joblib                      1.3.2
json5                       NA
jsonschema                  4.19.2
jsonschema_specifications   NA
jupyter_events              0.8.0
jupyter_server              2.10.0
jupyterlab_server           2.25.1
kiwisolver                  1.4.5
llvmlite                    0.41.1
markupsafe                  2.1.3
matplotlib_inline           0.1.6
mpl_toolkits                NA
natsort                     8.4.0
nbformat                    5.9.2
numba                       0.58.1
overrides                   NA
packaging                   23.2
parso                       0.8.3
patsy                       0.5.6
pexpect                     4.8.0
pickleshare                 0.7.5
pkg_resources               NA
platformdirs                4.1.0
prometheus_client           NA
prompt_toolkit              3.0.42
psutil                      5.9.0
ptyprocess                  0.7.0
pure_eval                   0.2.2
pycparser                   2.21
pydev_ipython               NA
pydevconsole                NA
pydevd                      2.9.5
pydevd_file_utils           NA
pydevd_plugins              NA
pydevd_tracing              NA
pygments                    2.17.2
pynndescent                 0.5.11
pyparsing                   3.1.1
pythonjsonlogger            NA
pytz                        2023.3.post1
referencing                 NA
requests                    2.31.0
rfc3339_validator           0.1.4
rfc3986_validator           0.1.1
rpds                        NA
scipy                       1.12.0
send2trash                  NA
six                         1.16.0
sklearn                     1.4.0
sniffio                     1.3.0
socks                       1.7.1
stack_data                  0.6.2
statsmodels                 0.14.1
threadpoolctl               3.2.0
tornado                     6.3.3
tqdm                        4.66.1
traitlets                   5.14.1
typing_extensions           NA
umap                        0.5.5
urllib3                     1.26.18
wcwidth                     0.2.13
websocket                   0.58.0
yaml                        6.0.1
zmq                         25.1.2
</pre>
</details> <!-- seems like this ends pre, so might as well be explicit -->
<pre>
-----
IPython             8.20.0
jupyter_client      8.6.0
jupyter_core        5.5.0
jupyterlab          4.0.8
notebook            7.0.6
-----
Python 3.11.7 (main, Dec 15 2023, 12:09:56) [Clang 14.0.6 ]
macOS-14.2.1-arm64-arm-64bit
-----
Session information updated at 2024-01-30 16:02
</pre>
</details>




```python

```

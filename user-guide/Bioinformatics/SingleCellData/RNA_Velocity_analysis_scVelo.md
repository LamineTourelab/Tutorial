# Getting Started with RNA Velocity


RNA velocity is based on bridging measurements to a underlying mechanism, mRNA splicing, with two modes indicating the current and future state [[RNA velocity—current challenges and future perspectives](https://www.embopress.org/doi/full/10.15252/msb.202110282)].  It is a method used to predict the future gene expression of a cell based on the measurement of both spliced and unspliced transcripts of mRNA [[2](https://towardsdatascience.com/rna-velocity-the-cells-internal-compass-cf8d75bb2f89)].

RNA velocity could be used to infer the direction of gene expression changes in single-cell RNA sequencing (scRNA-seq) data. It provides insights into the future state of individual cells by using the abundance of unspliced to spliced RNA transcripts. This ratio can indicate the transcriptional dynamics and potential fate of a cell, such as whether it is transitioning from one cell type to another or undergoing differentiation [[RNA velocity of single cells](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6130801/)].

+ **velocyto**

Velocyto is a package for the analysis of expression dynamics in single cell RNA seq data. In particular, it enables estimations of RNA velocities of single cells by distinguishing unspliced and spliced mRNAs in standard single-cell RNA sequencing protocols. It is the first paper proposed the concept of RNA velocity. velocyto predicted RNA velocity by solving the proposed differential equations for each gene [[RNA velocity of single cells](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6130801/)]. 

+ **scVelo**

scVelo is a method that solves the full transcriptional dynamics of splicing kinetics using a likelihood-based dynamical model. This generalizes RNA velocity to systems with transient cell states, which are common in development and in response to perturbations. scVelo was applied to disentangling subpopulation kinetics in neurogenesis and pancreatic endocrinogenesis. scVelo demonstrate the capabilities of the dynamical model on various cell lineages in hippocampal dentate gyrus neurogenesis and pancreatic endocrinogenesis [[Generalizing RNA velocity to transient cell states through dynamical modeling](https://www.nature.com/articles/s41587-020-0591-3)].

Here,I will go through the basics of scVelo. The following tutorials go straight into analysis of RNA velocity, latent time, driver identification and many more.

First of all, the input data for scVelo are two count matrices of pre-mature (unspliced) and mature (spliced) abundances, which were obtained from standard sequencing protocols, using the velocyto.


```python
from platform import python_version

print(python_version())
```


```python
#!pip install numpy==1.23.2 pandas==1.5.3 matplotlib==3.7.3 scanpy==1.9.6 igraph==0.9.8 scvelo==0.2.5 loompy==3.0.6 anndata==0.8.0
```


```python
#!pip install tqdm 
#!pip install ipywidgets
#!pip install pandas==1.1.5 
#!pip install numpy==1.21.1
```


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import scanpy as sc
import igraph
import scvelo as scv
import loompy as lmp
import anndata
import session_info
import warnings
warnings.filterwarnings('ignore')
```


```python
# Set parameters for plots, including size, color, etc.
scv.settings.verbosity = 3  # show errors(0), warnings(1), info(2), hints(3)
scv.settings.presenter_view = True  # set max width size for presenter view
scv.set_figure_params('scvelo')  # for beautified visualization
```


```python
adata = scv.read('/path/file.h5ad', cache=True)
adata
```


```python
adata.obs
```

+ Metadata info:
    The single-cell data consists of 8 different samples-A0 A6 B0 B6 C0 C6 DO D6:
    
    A- A549 WT Cells with no IFNb pretreatment.
    
    B- A549 WT Cells, pretreated with 10U of IFNb for 18 hours. 
    
    C- A549 IRF3 K.O Cells with no IFNb pretreatment.

    D- A549 IRF3 K.O Cells, pretreated with 10U of IFNb for 18 hours.

    0- a sample that was not stimulated with vRNA.

    6- a sample that was stimulated with vRNA for 6 hours.


```python
adata.obs['batch']
```

# Select only WT condition 


```python
# adata = adata[
#     adata.obs['condition'].isin([
#         'WT',
#         'WT&IFNb',
#         'WT&IFNb&vRNA',
#         'WT&vRNA'
#     ])
# ]
```


```python
adata
```


```python
%%time
ldata = scv.read('/path/file.loom', cache=True)
ldata
```


```python
scv.utils.clean_obs_names(adata)
scv.utils.clean_obs_names(ldata)
adata = scv.utils.merge(adata, ldata)
```


```python
adata.obs
```


```python
adata.layers
```


```python
scv.pl.proportions(adata, groupby="batch")
```


```python
#adata.obs['condition']=adata.obs['condition'].astype('category')
```

Here, the proportions of spliced/unspliced counts are displayed. Depending on the protocol used (Drop-Seq, Smart-Seq, inDrop and 10x Genomics Chromium protocols), we typically have between 10%-25% of unspliced molecules containing intronic sequences. We also advice you to examine the variations on cluster level to verify consistency in splicing efficiency. Here, we find variations, with slightly **lower** unspliced proportions at **Ciliated & Merkels cells**, then ***higher proportion*** at ***Mesangial & Keratinocytes cells***.


```python
scv.pl.proportions(adata, groupby="batch")
```


```python
sc.pl.embedding(adata, basis="umap", color= "batch", wspace=.3)
```

## Estimate RNA velocity

RNA velocity estimation can currently be tackled with three existing approaches:

• steady-state / deterministic model (using steady-state residuals)

• stochastic model (using second-order moments),

• dynamical model (using a likelihood-based framework).

 + **The steady-state / deterministic model:**, as being used in velocyto, estimates velocities as follows: Under the assumption that transcriptional phases (induction and repression) last sufficiently long to reach a steady-state equilibrium (active and inactive),**`velocities are quantified as the deviation of the observed ratio from its steady-state ratio`**. The equilibrium mRNA levels are approximated with a linear regression on the presumed steady states in the lower and upper quantiles. This simplification makes two fundamental assumptions: a common splicing rate across genes and steady-state mRNA levels to be reflected in the data. It can lead to errors in velocity estimates and cellular states as the assumptions are often violated, in particular when a population comprises multiple heterogeneous subpopulation dynamics.


 + **The stochastic model:** aims to better capture the steady states. By treating transcription, splicing and degradation as probabilistic events, the resulting Markov process is approximated by moment equations. By including secondorder moments, **`it exploits not only the balance of unspliced to spliced mRNA levels but also their covariation`**. It has been demonstrated on the endocrine pancreas that stochasticity adds valuable information, overall yielding higher consistency than the deterministic model, while remaining as efficient in computation time.


 + **The dynamical model:** (most powerful while computationally most expensive) solves the full dynamics of splicing kinetics for each gene. **`It thereby adapts RNA velocity to widely varying specifications such as non-stationary populations, as does not rely on the restrictions of a common splicing rate or steady states to be sampled`**.
 
     The splicing dynamics 
     
     `𝑑𝑢(𝑡)/𝑑𝑡 = 𝛼𝑘(𝑡) − 𝛽𝑢(𝑡), (4.1)` 
     
     `𝑑𝑠(𝑡)/𝑑𝑡 = 𝛽𝑢(𝑡) − 𝛾𝑠(4.2) (𝑡)`,
     
     is solved in a likelihood-based expectation-maximization framework, **`by iteratively estimating the parameters of reaction rates and latent cell-specific variables`**, i.e. transcriptional state k and cell-internal latent time t.It thereby aims to learn the unspliced/spliced phase trajectory. Four transcriptional states are modeled to account for all possible configurations of gene activity: two dynamic transient states (induction and repression) and two steady states (active and inactive) potentially reached after each dynamic transition.


```python
scv.pp.filter_genes(adata, min_shared_counts=20)
scv.pp.normalize_per_cell(adata)
scv.pp.filter_genes_dispersion(adata, n_top_genes=2000)
scv.pp.log1p(adata)
```


```python
scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
scv.pp.neighbors(adata)
scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
```


```python
%%time
scv.tl.recover_dynamics(adata, n_jobs=4)
```


```python
%%time
scv.tl.velocity(adata, mode='dynamical', n_jobs=4)
```


```python
%%time
scv.tl.velocity_graph(adata)
```


```python
adata
```

Running the dynamical model can take a while. Hence, you may want to store the results for re-use.


```python
adata.write('/Users/lamine/ShapiraLab/Analysis/Project2/Data/Project2&DynVelo.h5ad', compression='gzip')
```


```python
#adata = scv.read('/Users/lamine/ShapiraLab/Analysis/Project2/Data/Project2&DynVelo.h5ad', cache=True)
```


```python
adata
```


```python
print(adata.var['velocity_genes'].sum(), adata.n_vars)
top_genes = adata.var_names[adata.var.fit_likelihood.argsort()[::-1]]
scv.pl.scatter(adata, basis=top_genes[:10], ncols=4, color='batch', wspace=1.5, hspace=1, legend_loc='right margin')
```

There are 465 genes being used and 1102 cells.

## Project the velocities


```python
scv.pl.velocity_embedding_stream(adata, basis='umap', color='batch', 
                                 legend_loc='right margin')
```


```python
scv.pl.velocity_embedding_grid(adata, basis='umap', color='batch', 
                          legend_loc='right margin')
```

## Interprete the velocities

See the gif <a href="https://user-images.githubusercontent.com/31883718/80227452-eb822480-864d-11ea-9399-56886c5e2785.gif">here</a> to get an idea of how to interpret a spliced vs. unspliced phase portrait. Gene activity is orchestrated by transcriptional regulation. Transcriptional induction for a particular gene results in an **increase of (newly transcribed) precursor unspliced mRNAs** while, conversely, repression or absence of transcription results in a **decrease of unspliced mRNAs**. Spliced mRNAs is produced from unspliced mRNA and follows the same trend with a time lag. Time is a hidden/latent variable. Thus, the dynamics needs to be inferred from what is actually measured: spliced and unspliced mRNAs as displayed in the phase portrait.


```python
adata
```


```python
df = scv.get_df(adata, 'rank_genes_groups/names')
df.head(10)
```


```python
scv.pl.velocity(adata, ['AC090498.1',  'HLA-B', 'PHLDA2', 'FBXO32', 'IFIT2','EEEF1A1'], color='batch')
```

## Identify important genes
 
 + **By Condition**


```python
scv.tl.rank_velocity_genes(adata, groupby='batch', min_corr=.3)

df = scv.DataFrame(adata.uns['rank_velocity_genes']['names'])
df.head(30)
```


```python
for condition in ['SS001', 'SS002', 'SS003', 'SS004']:
    scv.pl.scatter(adata, df[condition][:5], ylabel=condition, color='batch', wspace=.6)
```

## Speed and coherence

Two more useful stats: - The speed or rate of differentiation is given by the length of the velocity vector. - The coherence of the vector field (i.e., how a velocity vector correlates with its neighboring velocities) provides a measure of confidence.


```python
scv.tl.velocity_confidence(adata)
keys = 'velocity_length', 'velocity_confidence'
scv.pl.scatter(adata, c=keys, cmap='coolwarm', perc=[5, 95])
```


```python
df = adata.obs.groupby('batch')[keys].mean().T
df.style.background_gradient(cmap='coolwarm', axis=1)
```

## Velocity graph and pseudotime

We can visualize the velocity graph to portray all velocity-inferred cell-to-cell connections/transitions. It can be confined to high-probability transitions by setting a threshold. 


```python
scv.pl.velocity_graph(adata, threshold=.9, color='batch', 
                          legend_loc='right margin')
```


```python
#x, y = scv.utils.get_cell_transitions(adata, basis='umap', n_neighbors=10, starting_cell=70)
#ax = scv.pl.velocity_graph(adata, c='lightgrey', edge_width=.05, show=False)
#ax = scv.pl.scatter(adata, x=x, y=y, s=120, c='ascending', cmap='gnuplot', ax=ax)
```

Finally, based on the velocity graph, a velocity pseudotime can be computed. After inferring a distribution over root cells from the graph, it measures the average number of steps it takes to reach a cell after walking along the graph starting from the root cells.


```python
scv.tl.velocity_pseudotime(adata)
scv.pl.scatter(adata, color='velocity_pseudotime', cmap='gnuplot')
```

## PAGA velocity graph

PAGA graph abstraction has benchmarked as top-performing method for trajectory inference. It provides a graph-like map of the data topology with weighted edges corresponding to the connectivity between two clusters. Here, PAGA is extended by velocity-inferred directionality.


```python
adata
```


```python
# PAGA requires to install igraph, if not done yet.
!pip install python-igraph --upgrade --quiet
```


```python
# this is needed due to a current bug - bugfix is coming soon.
adata.uns['neighbors']['distances'] = adata.obsp['distances']
adata.uns['neighbors']['connectivities'] = adata.obsp['connectivities']

scv.tl.paga(adata, groups='batch', use_time_prior=False)
df = scv.get_df(adata, 'paga/transitions_confidence', precision=2).T
df.style.background_gradient(cmap='Blues').format('{:.2g}')
```

This reads from left/row to right/column, thus e.g. assigning a confident transition from Merkel sells to Basophils.

This table can be summarized by a directed graph superimposed onto the UMAP embedding.


```python
scv.pl.paga(adata, basis='umap', size=50, alpha=.2, 
            min_edge_width=2, node_size_scale=1)
```


```python
scv.pl.velocity_embedding_stream(adata, basis='umap', color='batch', 
                                 legend_loc='right margin', ncols=1)
```

Here we observb **2** main velocity direction, one in **Gamma delta T cells and Monocytes**. 


```python
results_file = '/path/file&DynVelo.h5ad'  
adata.write(results_file)
```

## Some more analysis for dynamical mode

### Kinetic rate paramters

The rates of RNA transcription, splicing and degradation are estimated without the need of any experimental data.

They can be useful to better understand the cell identity and phenotypic heterogeneity.


```python
df = adata.var
df = df[(df['fit_likelihood'] > .1) & df['velocity_genes'] == True]

kwargs = dict(xscale='log', fontsize=16)
with scv.GridSpec(ncols=3) as pl:
    pl.hist(df['fit_alpha'], xlabel='transcription rate', **kwargs)
    pl.hist(df['fit_beta'] * df['fit_scaling'], xlabel='splicing rate', xticks=[.1, .4, 1], **kwargs)
    pl.hist(df['fit_gamma'], xlabel='degradation rate', xticks=[.1, .4, 1], **kwargs)

scv.get_df(adata, 'fit*', dropna=True).head()
```

The estimated gene-specific parameters comprise rates of transription (fit_alpha), splicing (fit_beta), degradation (fit_gamma), switching time point (fit_t_), a scaling parameter to adjust for under-represented unspliced reads (fit_scaling), standard deviation of unspliced and spliced reads (fit_std_u, fit_std_s), the gene likelihood (fit_likelihood), inferred steady-state levels (fit_steady_u, fit_steady_s) with their corresponding p-values (fit_pval_steady_u, fit_pval_steady_s), the overall model variance (fit_variance), and a scaling factor to align the gene-wise latent times to a universal, gene-shared latent time (fit_alignment_scaling).



### Latent time

The dynamical model recovers the latent time of the underlying cellular processes. This latent time represents the cell’s internal clock and approximates the real time experienced by cells as they differentiate, based only on its transcriptional dynamics.


```python
scv.tl.latent_time(adata)
scv.pl.scatter(adata, color='latent_time', color_map='gnuplot', size=80)
```


```python
top_genes = adata.var['fit_likelihood'].sort_values(ascending=False).index[:300]
scv.pl.heatmap(adata, var_names=top_genes, sortby='latent_time', col_color='batch', n_convolve=100)
```

### Top-likelihood genes

Driver genes display pronounced dynamic behavior and are systematically detected via their characterization by high likelihoods in the dynamic model.


```python
top_genes = adata.var['fit_likelihood'].sort_values(ascending=False).index
scv.pl.scatter(adata, basis=top_genes[:15], ncols=5, frameon=False, color='batch')
```

### Cluster-specific top-likelihood genes

Moreover, partial gene likelihoods can be computed for a each cluster of cells to enable cluster-specific identification of potential drivers.


```python
scv.tl.rank_dynamical_genes(adata, groupby='batch')
df = scv.get_df(adata, 'rank_dynamical_genes/names')
df.head(30)
```


```python
for condition in ['SS001', 'SS002', 'SS003', 'SS004', 'SS005', 'SS006', 'SS007', 'SS008' ]:
    scv.pl.scatter(adata, df[condition][:5], ylabel=condition, color='batch', wspace=.6)
```


```python

```


```python
session_info.show()
```




<details>
<summary>Click to view session information</summary>
<pre>
-----
anndata             0.10.4
igraph              0.11.3
loompy              3.0.6
matplotlib          3.7.3
numpy               1.23.2
pandas              1.4.1
scanpy              1.9.6
scvelo              0.2.5
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
importlib_metadata          NA
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
leidenalg                   0.10.2
llvmlite                    0.41.1
markupsafe                  2.1.3
matplotlib_inline           0.1.6
mpl_toolkits                NA
natsort                     8.4.0
nbformat                    5.9.2
numba                       0.58.1
numpy_groupies              0.10.2
overrides                   NA
packaging                   23.2
parso                       0.8.3
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
texttable                   1.7.0
threadpoolctl               3.2.0
tornado                     6.3.3
traitlets                   5.14.1
typing_extensions           NA
urllib3                     1.26.18
wcwidth                     0.2.13
websocket                   0.58.0
yaml                        6.0.1
zipp                        NA
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
Python 3.11.7 | packaged by conda-forge | (main, Dec 23 2023, 14:38:07) [Clang 16.0.6 ]
macOS-14.2.1-arm64-arm-64bit
-----
Session information updated at 2024-01-30 14:58
</pre>
</details>




```python

```

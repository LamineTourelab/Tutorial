# Explain tensorflow model using Shap and Model Weights on METABRIC data


```python
from platform import python_version
python_version()
```




    '3.8.18'




```python
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential

# Check for TensorFlow GPU access
print(f"TensorFlow has access to the following devices:\n{tf.config.list_physical_devices()}")

# See TensorFlow version
print(f"TensorFlow version: {tf.__version__}")
```

    TensorFlow has access to the following devices:
    [PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'), PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
    TensorFlow version: 2.13.0



```python
tf.test.is_gpu_available()
```

    WARNING:tensorflow:From /var/folders/r6/gdf2rxfj40dc6v2l0101z47m0000gn/T/ipykernel_38703/337460670.py:1: is_gpu_available (from tensorflow.python.framework.test_util) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.config.list_physical_devices('GPU')` instead.


    2024-02-19 13:40:35.802602: I metal_plugin/src/device/metal_device.cc:1154] Metal device set to: Apple M3 Max
    2024-02-19 13:40:35.802618: I metal_plugin/src/device/metal_device.cc:296] systemMemory: 128.00 GB
    2024-02-19 13:40:35.802622: I metal_plugin/src/device/metal_device.cc:313] maxCacheSize: 48.00 GB
    2024-02-19 13:40:35.802646: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:303] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
    2024-02-19 13:40:35.802662: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:269] Created TensorFlow device (/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)





    True



# Data load and preprocessing 

## Expression data


```python
df = pd.read_csv("/Users/lamine/Explainqble AI /metabric_test.csv")
df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CD52</th>
      <th>DARC</th>
      <th>DCN</th>
      <th>DB005376</th>
      <th>TAT</th>
      <th>GSTM1</th>
      <th>UGT2B11</th>
      <th>AKR7A3</th>
      <th>SERHL2</th>
      <th>ASS1</th>
      <th>...</th>
      <th>MYB</th>
      <th>PROM1</th>
      <th>GSTT1</th>
      <th>NELL2</th>
      <th>CST5</th>
      <th>CCL5</th>
      <th>TFF3</th>
      <th>CDH3</th>
      <th>SLC39A6</th>
      <th>SHISA2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8.240128</td>
      <td>10.731211</td>
      <td>11.251592</td>
      <td>5.350604</td>
      <td>5.698745</td>
      <td>5.626606</td>
      <td>5.845062</td>
      <td>8.334491</td>
      <td>7.150713</td>
      <td>9.887783</td>
      <td>...</td>
      <td>7.864506</td>
      <td>10.475799</td>
      <td>5.236212</td>
      <td>6.462909</td>
      <td>5.333817</td>
      <td>8.771015</td>
      <td>10.545305</td>
      <td>8.588759</td>
      <td>8.287300</td>
      <td>6.155340</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.441887</td>
      <td>6.498731</td>
      <td>9.968656</td>
      <td>5.701508</td>
      <td>5.416231</td>
      <td>5.108180</td>
      <td>5.382890</td>
      <td>10.277779</td>
      <td>6.070879</td>
      <td>6.203103</td>
      <td>...</td>
      <td>10.699097</td>
      <td>5.977531</td>
      <td>8.450049</td>
      <td>7.486917</td>
      <td>5.464502</td>
      <td>8.216436</td>
      <td>10.422146</td>
      <td>5.838056</td>
      <td>10.380559</td>
      <td>9.409817</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.977708</td>
      <td>6.615727</td>
      <td>9.632207</td>
      <td>6.346358</td>
      <td>5.480066</td>
      <td>5.356168</td>
      <td>7.798285</td>
      <td>9.117568</td>
      <td>6.230590</td>
      <td>7.928613</td>
      <td>...</td>
      <td>9.861437</td>
      <td>8.517411</td>
      <td>7.230715</td>
      <td>11.957439</td>
      <td>5.359362</td>
      <td>8.012079</td>
      <td>12.201802</td>
      <td>6.681570</td>
      <td>10.009376</td>
      <td>9.094121</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8.045781</td>
      <td>5.806614</td>
      <td>8.927632</td>
      <td>5.628718</td>
      <td>5.746114</td>
      <td>5.402901</td>
      <td>6.043053</td>
      <td>10.057702</td>
      <td>11.682904</td>
      <td>10.047193</td>
      <td>...</td>
      <td>9.138474</td>
      <td>9.099391</td>
      <td>8.072639</td>
      <td>12.478907</td>
      <td>5.523048</td>
      <td>9.245577</td>
      <td>14.169804</td>
      <td>6.392376</td>
      <td>11.141299</td>
      <td>10.039994</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9.001653</td>
      <td>7.928994</td>
      <td>9.356798</td>
      <td>5.484226</td>
      <td>5.152513</td>
      <td>5.401268</td>
      <td>8.511554</td>
      <td>11.127156</td>
      <td>7.472530</td>
      <td>7.200276</td>
      <td>...</td>
      <td>9.591358</td>
      <td>7.264378</td>
      <td>8.975517</td>
      <td>10.044922</td>
      <td>5.034380</td>
      <td>10.243518</td>
      <td>13.568835</td>
      <td>8.476834</td>
      <td>8.916101</td>
      <td>5.929184</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 295 columns</p>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1897 entries, 0 to 1896
    Columns: 295 entries, CD52 to SHISA2
    dtypes: float64(295)
    memory usage: 4.3 MB



```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CD52</th>
      <th>DARC</th>
      <th>DCN</th>
      <th>DB005376</th>
      <th>TAT</th>
      <th>GSTM1</th>
      <th>UGT2B11</th>
      <th>AKR7A3</th>
      <th>SERHL2</th>
      <th>ASS1</th>
      <th>...</th>
      <th>MYB</th>
      <th>PROM1</th>
      <th>GSTT1</th>
      <th>NELL2</th>
      <th>CST5</th>
      <th>CCL5</th>
      <th>TFF3</th>
      <th>CDH3</th>
      <th>SLC39A6</th>
      <th>SHISA2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1897.000000</td>
      <td>1897.000000</td>
      <td>1897.000000</td>
      <td>1897.000000</td>
      <td>1897.000000</td>
      <td>1897.000000</td>
      <td>1897.000000</td>
      <td>1897.000000</td>
      <td>1897.000000</td>
      <td>1897.000000</td>
      <td>...</td>
      <td>1897.000000</td>
      <td>1897.000000</td>
      <td>1897.000000</td>
      <td>1897.000000</td>
      <td>1897.000000</td>
      <td>1897.000000</td>
      <td>1897.000000</td>
      <td>1897.000000</td>
      <td>1897.000000</td>
      <td>1897.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>8.522002</td>
      <td>7.439279</td>
      <td>8.592254</td>
      <td>6.084079</td>
      <td>6.267616</td>
      <td>6.477882</td>
      <td>6.920908</td>
      <td>9.397352</td>
      <td>7.558455</td>
      <td>8.298495</td>
      <td>...</td>
      <td>9.743111</td>
      <td>8.041666</td>
      <td>8.295523</td>
      <td>7.466347</td>
      <td>6.033271</td>
      <td>9.845330</td>
      <td>11.742209</td>
      <td>7.465389</td>
      <td>9.204424</td>
      <td>7.725656</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.349624</td>
      <td>1.323882</td>
      <td>1.366120</td>
      <td>1.489150</td>
      <td>1.623607</td>
      <td>1.490238</td>
      <td>2.132190</td>
      <td>1.280389</td>
      <td>1.724598</td>
      <td>1.314099</td>
      <td>...</td>
      <td>1.242550</td>
      <td>1.996117</td>
      <td>1.691650</td>
      <td>1.532031</td>
      <td>1.500256</td>
      <td>1.357065</td>
      <td>2.444823</td>
      <td>1.274105</td>
      <td>1.620264</td>
      <td>1.659966</td>
    </tr>
    <tr>
      <th>min</th>
      <td>5.018810</td>
      <td>5.099984</td>
      <td>5.074217</td>
      <td>4.922326</td>
      <td>4.925973</td>
      <td>4.939510</td>
      <td>4.988302</td>
      <td>6.888636</td>
      <td>5.214098</td>
      <td>5.001618</td>
      <td>...</td>
      <td>5.565536</td>
      <td>5.047322</td>
      <td>4.854543</td>
      <td>5.030010</td>
      <td>4.965204</td>
      <td>5.685101</td>
      <td>5.154748</td>
      <td>5.103031</td>
      <td>5.510203</td>
      <td>5.119337</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>7.526147</td>
      <td>6.337077</td>
      <td>7.585572</td>
      <td>5.315275</td>
      <td>5.400663</td>
      <td>5.428807</td>
      <td>5.547688</td>
      <td>8.359180</td>
      <td>6.265815</td>
      <td>7.277712</td>
      <td>...</td>
      <td>9.072006</td>
      <td>6.297426</td>
      <td>7.469392</td>
      <td>6.264153</td>
      <td>5.337878</td>
      <td>8.875585</td>
      <td>10.657896</td>
      <td>6.461509</td>
      <td>7.869267</td>
      <td>6.363869</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>8.448275</td>
      <td>7.331663</td>
      <td>8.608817</td>
      <td>5.461374</td>
      <td>5.563156</td>
      <td>5.624529</td>
      <td>5.881415</td>
      <td>9.331409</td>
      <td>7.083379</td>
      <td>8.280220</td>
      <td>...</td>
      <td>10.023695</td>
      <td>7.623121</td>
      <td>8.889979</td>
      <td>7.056264</td>
      <td>5.484401</td>
      <td>9.857851</td>
      <td>12.473404</td>
      <td>7.303850</td>
      <td>9.201048</td>
      <td>7.358426</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>9.428863</td>
      <td>8.370030</td>
      <td>9.566763</td>
      <td>5.971988</td>
      <td>6.175448</td>
      <td>7.490048</td>
      <td>7.556015</td>
      <td>10.241203</td>
      <td>8.371308</td>
      <td>9.256413</td>
      <td>...</td>
      <td>10.654395</td>
      <td>9.607842</td>
      <td>9.489065</td>
      <td>8.371956</td>
      <td>5.818663</td>
      <td>10.791775</td>
      <td>13.588736</td>
      <td>8.255375</td>
      <td>10.508201</td>
      <td>8.869039</td>
    </tr>
    <tr>
      <th>max</th>
      <td>13.374739</td>
      <td>11.619202</td>
      <td>12.478475</td>
      <td>13.010996</td>
      <td>13.166804</td>
      <td>12.070735</td>
      <td>14.145451</td>
      <td>13.512971</td>
      <td>13.731721</td>
      <td>12.182876</td>
      <td>...</td>
      <td>12.091906</td>
      <td>13.569006</td>
      <td>12.784519</td>
      <td>13.110442</td>
      <td>13.922840</td>
      <td>14.004198</td>
      <td>14.808641</td>
      <td>12.003642</td>
      <td>13.440167</td>
      <td>12.874823</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 295 columns</p>
</div>



## label data 


```python
metadata = pd.read_csv("/Users/lamine/Explainqble AI /metabric_clin.csv")
metadata.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PATIENT_ID</th>
      <th>LYMPH_NODES_EXAMINED_POSITIVE</th>
      <th>NPI</th>
      <th>CELLULARITY</th>
      <th>CHEMOTHERAPY</th>
      <th>COHORT</th>
      <th>ER_IHC</th>
      <th>HER2_SNP6</th>
      <th>HORMONE_THERAPY</th>
      <th>INFERRED_MENOPAUSAL_STATE</th>
      <th>...</th>
      <th>OS_STATUS</th>
      <th>CLAUDIN_SUBTYPE</th>
      <th>THREEGENE</th>
      <th>VITAL_STATUS</th>
      <th>LATERALITY</th>
      <th>RADIO_THERAPY</th>
      <th>HISTOLOGICAL_SUBTYPE</th>
      <th>BREAST_SURGERY</th>
      <th>RFS_STATUS</th>
      <th>RFS_MONTHS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MB-0000</td>
      <td>10.0</td>
      <td>6.044</td>
      <td>NaN</td>
      <td>NO</td>
      <td>1.0</td>
      <td>Positve</td>
      <td>NEUTRAL</td>
      <td>YES</td>
      <td>Post</td>
      <td>...</td>
      <td>0:LIVING</td>
      <td>claudin-low</td>
      <td>ER-/HER2-</td>
      <td>Living</td>
      <td>Right</td>
      <td>YES</td>
      <td>Ductal/NST</td>
      <td>MASTECTOMY</td>
      <td>0:Not Recurred</td>
      <td>138.65</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MB-0002</td>
      <td>0.0</td>
      <td>4.020</td>
      <td>High</td>
      <td>NO</td>
      <td>1.0</td>
      <td>Positve</td>
      <td>NEUTRAL</td>
      <td>YES</td>
      <td>Pre</td>
      <td>...</td>
      <td>0:LIVING</td>
      <td>LumA</td>
      <td>ER+/HER2- High Prolif</td>
      <td>Living</td>
      <td>Right</td>
      <td>YES</td>
      <td>Ductal/NST</td>
      <td>BREAST CONSERVING</td>
      <td>0:Not Recurred</td>
      <td>83.52</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MB-0005</td>
      <td>1.0</td>
      <td>4.030</td>
      <td>High</td>
      <td>YES</td>
      <td>1.0</td>
      <td>Positve</td>
      <td>NEUTRAL</td>
      <td>YES</td>
      <td>Pre</td>
      <td>...</td>
      <td>1:DECEASED</td>
      <td>LumB</td>
      <td>NaN</td>
      <td>Died of Disease</td>
      <td>Right</td>
      <td>NO</td>
      <td>Ductal/NST</td>
      <td>MASTECTOMY</td>
      <td>1:Recurred</td>
      <td>151.28</td>
    </tr>
    <tr>
      <th>3</th>
      <td>MB-0006</td>
      <td>3.0</td>
      <td>4.050</td>
      <td>Moderate</td>
      <td>YES</td>
      <td>1.0</td>
      <td>Positve</td>
      <td>NEUTRAL</td>
      <td>YES</td>
      <td>Pre</td>
      <td>...</td>
      <td>0:LIVING</td>
      <td>LumB</td>
      <td>NaN</td>
      <td>Living</td>
      <td>Right</td>
      <td>YES</td>
      <td>Mixed</td>
      <td>MASTECTOMY</td>
      <td>0:Not Recurred</td>
      <td>162.76</td>
    </tr>
    <tr>
      <th>4</th>
      <td>MB-0008</td>
      <td>8.0</td>
      <td>6.080</td>
      <td>High</td>
      <td>YES</td>
      <td>1.0</td>
      <td>Positve</td>
      <td>NEUTRAL</td>
      <td>YES</td>
      <td>Post</td>
      <td>...</td>
      <td>1:DECEASED</td>
      <td>LumB</td>
      <td>ER+/HER2- High Prolif</td>
      <td>Died of Disease</td>
      <td>Right</td>
      <td>YES</td>
      <td>Mixed</td>
      <td>MASTECTOMY</td>
      <td>1:Recurred</td>
      <td>18.55</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>




```python
metadata.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1897 entries, 0 to 1896
    Data columns (total 24 columns):
     #   Column                         Non-Null Count  Dtype  
    ---  ------                         --------------  -----  
     0   PATIENT_ID                     1897 non-null   object 
     1   LYMPH_NODES_EXAMINED_POSITIVE  1897 non-null   float64
     2   NPI                            1897 non-null   float64
     3   CELLULARITY                    1843 non-null   object 
     4   CHEMOTHERAPY                   1897 non-null   object 
     5   COHORT                         1897 non-null   float64
     6   ER_IHC                         1867 non-null   object 
     7   HER2_SNP6                      1897 non-null   object 
     8   HORMONE_THERAPY                1897 non-null   object 
     9   INFERRED_MENOPAUSAL_STATE      1897 non-null   object 
     10  SEX                            1897 non-null   object 
     11  INTCLUST                       1897 non-null   object 
     12  AGE_AT_DIAGNOSIS               1897 non-null   float64
     13  OS_MONTHS                      1897 non-null   float64
     14  OS_STATUS                      1897 non-null   object 
     15  CLAUDIN_SUBTYPE                1897 non-null   object 
     16  THREEGENE                      1694 non-null   object 
     17  VITAL_STATUS                   1897 non-null   object 
     18  LATERALITY                     1792 non-null   object 
     19  RADIO_THERAPY                  1897 non-null   object 
     20  HISTOLOGICAL_SUBTYPE           1882 non-null   object 
     21  BREAST_SURGERY                 1875 non-null   object 
     22  RFS_STATUS                     1897 non-null   object 
     23  RFS_MONTHS                     1897 non-null   float64
    dtypes: float64(6), object(18)
    memory usage: 355.8+ KB



```python
metadata.columns
```




    Index(['PATIENT_ID', 'LYMPH_NODES_EXAMINED_POSITIVE', 'NPI', 'CELLULARITY',
           'CHEMOTHERAPY', 'COHORT', 'ER_IHC', 'HER2_SNP6', 'HORMONE_THERAPY',
           'INFERRED_MENOPAUSAL_STATE', 'SEX', 'INTCLUST', 'AGE_AT_DIAGNOSIS',
           'OS_MONTHS', 'OS_STATUS', 'CLAUDIN_SUBTYPE', 'THREEGENE',
           'VITAL_STATUS', 'LATERALITY', 'RADIO_THERAPY', 'HISTOLOGICAL_SUBTYPE',
           'BREAST_SURGERY', 'RFS_STATUS', 'RFS_MONTHS'],
          dtype='object')




```python
print(f"The total patient ids are {metadata['PATIENT_ID'].count()}, from those the unique ids are {metadata['PATIENT_ID'].value_counts().shape[0]} ")
```

    The total patient ids are 1897, from those the unique ids are 1897 



```python
columns = metadata.keys()
columns = list(columns)
print(columns)
```

    ['PATIENT_ID', 'LYMPH_NODES_EXAMINED_POSITIVE', 'NPI', 'CELLULARITY', 'CHEMOTHERAPY', 'COHORT', 'ER_IHC', 'HER2_SNP6', 'HORMONE_THERAPY', 'INFERRED_MENOPAUSAL_STATE', 'SEX', 'INTCLUST', 'AGE_AT_DIAGNOSIS', 'OS_MONTHS', 'OS_STATUS', 'CLAUDIN_SUBTYPE', 'THREEGENE', 'VITAL_STATUS', 'LATERALITY', 'RADIO_THERAPY', 'HISTOLOGICAL_SUBTYPE', 'BREAST_SURGERY', 'RFS_STATUS', 'RFS_MONTHS']



```python
# Remove unnecesary elements
columns.remove('PATIENT_ID')
# Get the total classes
print(f"There are {len(columns)} columns of labels for these conditions: {columns}")
```

    There are 23 columns of labels for these conditions: ['LYMPH_NODES_EXAMINED_POSITIVE', 'NPI', 'CELLULARITY', 'CHEMOTHERAPY', 'COHORT', 'ER_IHC', 'HER2_SNP6', 'HORMONE_THERAPY', 'INFERRED_MENOPAUSAL_STATE', 'SEX', 'INTCLUST', 'AGE_AT_DIAGNOSIS', 'OS_MONTHS', 'OS_STATUS', 'CLAUDIN_SUBTYPE', 'THREEGENE', 'VITAL_STATUS', 'LATERALITY', 'RADIO_THERAPY', 'HISTOLOGICAL_SUBTYPE', 'BREAST_SURGERY', 'RFS_STATUS', 'RFS_MONTHS']



```python
metadata['THREEGENE'].unique()
```




    array(['ER-/HER2-', 'ER+/HER2- High Prolif', nan, 'ER+/HER2- Low Prolif',
           'HER2+'], dtype=object)




```python
print(f"The total patient ids are {metadata['PATIENT_ID'].count()}, from those the unique ids CHEMOTHERAPY are {metadata['CHEMOTHERAPY'].value_counts().shape[0]} ")
```

    The total patient ids are 1897, from those the unique ids CHEMOTHERAPY are 2 



```python
print(f'Number of patient ER-/HER2- :%i'  %metadata[metadata['THREEGENE']=='ER-/HER2-'].count()[0])
print(f'Number of patient ER+/HER2- High Prolif :%i'  %metadata[metadata['THREEGENE']=='ER+/HER2- High Prolif'].count()[0])
print(f'Number of patient ER+/HER2- Low Prolif :%i'  %metadata[metadata['THREEGENE']=='ER+/HER2- Low Prolif'].count()[0])
print(f'Number of patient HER2+ :%i'  %metadata[metadata['THREEGENE']=='HER2+'].count()[0])
print(f'Number of patient nan :%i'  %metadata[metadata['THREEGENE']=='nan'].count()[0])
```

    Number of patient ER-/HER2- :290
    Number of patient ER+/HER2- High Prolif :598
    Number of patient ER+/HER2- Low Prolif :618
    Number of patient HER2+ :188
    Number of patient nan :0



```python
plt.figure(figsize=(8,5), dpi=100)

plt.style.use('ggplot')

ER = metadata[metadata['THREEGENE']=='ER-/HER2-'].count()[0]
ERhigh = metadata[metadata['THREEGENE']=='ER+/HER2- High Prolif'].count()[0]
ERlow = metadata[metadata['THREEGENE']=='ER+/HER2- Low Prolif'].count()[0]
HER2 = metadata[metadata['THREEGENE']=='HER2+'].count()[0]
nan = metadata[metadata['THREEGENE']=='nan'].count()[0]


Condition = [ER, ERhigh, ERlow, HER2, nan]
label = ['ER', 'ERhigh', 'ERlow', 'HER2', 'nan']
explode = (0,0,0,.4,0)

plt.title('Pourcentage of number of patient per condition')

plt.pie(Condition, labels=label, explode=explode, pctdistance=0.8,autopct='%.2f %%')
plt.show()
```


    
![png](output_19_0.png)
    



```python
print(f'Number of patient CHEMOTHERAPY Yes :%i'  %metadata[metadata['CHEMOTHERAPY']=='YES'].count()[0])
print(f'Number of patient CHEMOTHERAPY NO :%i'  %metadata[metadata['CHEMOTHERAPY']=='NO'].count()[0])

```

    Number of patient CHEMOTHERAPY Yes :396
    Number of patient CHEMOTHERAPY NO :1501



```python
Outcome = pd.DataFrame(metadata['CHEMOTHERAPY'])
Outcome.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CHEMOTHERAPY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NO</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NO</td>
    </tr>
    <tr>
      <th>2</th>
      <td>YES</td>
    </tr>
    <tr>
      <th>3</th>
      <td>YES</td>
    </tr>
    <tr>
      <th>4</th>
      <td>YES</td>
    </tr>
  </tbody>
</table>
</div>




```python
Outcome = Outcome.replace("YES",1)
Outcome = Outcome.replace("NO",0)
Outcome.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CHEMOTHERAPY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('Labels counts in Outcome Yes and No respectively:', np.bincount(Outcome['CHEMOTHERAPY']))
```

    Labels counts in Outcome Yes and No respectively: [1501  396]


here we clearly dealing with class imbalance.


```python
df.index = metadata['PATIENT_ID']
Outcome.index = metadata['PATIENT_ID']
```


```python
Outcome = Outcome[Outcome.index.isin(df.index)]
Outcome.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CHEMOTHERAPY</th>
    </tr>
    <tr>
      <th>PATIENT_ID</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MB-0000</th>
      <td>0</td>
    </tr>
    <tr>
      <th>MB-0002</th>
      <td>0</td>
    </tr>
    <tr>
      <th>MB-0005</th>
      <td>1</td>
    </tr>
    <tr>
      <th>MB-0006</th>
      <td>1</td>
    </tr>
    <tr>
      <th>MB-0008</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



here we clearly dealing with class imbalance.
### Class imbalance correct using imblearn


```python
#!pip install -U imbalanced-learn
from imblearn.over_sampling import RandomOverSampler
```


```python
ros = RandomOverSampler(random_state=0)
df_resampled, Outcome_resampled = ros.fit_resample(df, Outcome['CHEMOTHERAPY'])
```


```python
from collections import Counter
print(sorted(Counter(Outcome_resampled).items()))
```

    [(0, 1501), (1, 1501)]



```python
print('Labels counts in Outcome are now:', np.bincount(Outcome_resampled))
```

    Labels counts in Outcome are now: [1501 1501]


## Data preparation for deep learning model


```python
df_resampled = df_resampled.astype('float32')
Outcome_resampled = Outcome_resampled.astype('float32')
```


```python
X_train, X_test, Y_train, Y_test = train_test_split(df_resampled, Outcome_resampled,
                                                    test_size=0.3, random_state=42, shuffle=True)
```


```python
print(f"X_train.shape = {X_train.shape}")
print(f"X_train.dtypes = {X_train.dtypes}")
```

    X_train.shape = (2101, 295)
    X_train.dtypes = CD52        float32
    DARC        float32
    DCN         float32
    DB005376    float32
    TAT         float32
                 ...   
    CCL5        float32
    TFF3        float32
    CDH3        float32
    SLC39A6     float32
    SHISA2      float32
    Length: 295, dtype: object



```python
x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, 
                                                  shuffle=True)
```


```python
scaler  = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)   
x_test_scaled = scaler.transform(X_test)
```


```python
print(f"x_train_scaled.shape = {x_train_scaled.shape}")
print(f"x_val_scaled.shape = {x_val_scaled.shape}")
print(f"x_test_scaled.shape = {x_test_scaled.shape}")
```

    x_train_scaled.shape = (1575, 295)
    x_val_scaled.shape = (526, 295)
    x_test_scaled.shape = (901, 295)



```python
# y_train = np.asarray(Y_train).astype('float32').reshape((-1,1))
```

# Model


```python
EPOCHS = 50
NB_CLASSES = 1   
DROPOUT = 0.3
```


```python
# The model hyperparatmeter was defined with gridSearchCV using scikeras.wrappers below.
model = Sequential()
model.add(Dense(10, input_shape=x_train_scaled.shape[1:], activation = 'relu', name='dense_layer'))
model.add(Dropout(DROPOUT))
model.add(Dense(10, activation = 'relu', name='dense_layer_2'))
model.add(Dropout(DROPOUT))
model.add(Dense(NB_CLASSES, activation= "sigmoid", name='dense_layer_4'))                              
```


```python
model.summary()
```

    Model: "sequential_96"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense_layer (Dense)         (None, 10)                2960      
                                                                     
     dropout_56 (Dropout)        (None, 10)                0         
                                                                     
     dense_layer_2 (Dense)       (None, 10)                110       
                                                                     
     dropout_57 (Dropout)        (None, 10)                0         
                                                                     
     dense_layer_4 (Dense)       (None, 1)                 11        
                                                                     
    =================================================================
    Total params: 3081 (12.04 KB)
    Trainable params: 3081 (12.04 KB)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________



```python
keras.utils.plot_model(model, show_shapes=True)
```




    
![png](output_44_0.png)
    




```python
early_stopping = EarlyStopping(
    patience  = 10,     # how many epochs to wait before stopping
    restore_best_weights=True,
)
```


```python
model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])
```

    WARNING:absl:At this time, the v2.11+ optimizer `tf.keras.optimizers.Adam` runs slowly on M1/M2 Macs, please use the legacy Keras optimizer instead, located at `tf.keras.optimizers.legacy.Adam`.
    WARNING:absl:There is a known slowdown when using v2.11+ Keras optimizers on M1/M2 Macs. Falling back to the legacy Keras optimizer, i.e., `tf.keras.optimizers.legacy.Adam`.



```python
#training the moodel
results = model.fit(x_train_scaled, y_train, epochs=EPOCHS, 
                    verbose=1, callbacks=[early_stopping],
                   validation_data=(x_val_scaled, y_val))
```

    Epoch 1/50


    2024-02-20 12:33:46.256299: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type GPU is enabled.


    50/50 [==============================] - ETA: 0s - loss: 0.6839 - accuracy: 0.5829

    2024-02-20 12:33:47.283473: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type GPU is enabled.


    50/50 [==============================] - 2s 21ms/step - loss: 0.6839 - accuracy: 0.5829 - val_loss: 0.5990 - val_accuracy: 0.6768
    Epoch 2/50
    50/50 [==============================] - 0s 8ms/step - loss: 0.6034 - accuracy: 0.6819 - val_loss: 0.5764 - val_accuracy: 0.6977
    Epoch 3/50
    50/50 [==============================] - 0s 7ms/step - loss: 0.6008 - accuracy: 0.6832 - val_loss: 0.5729 - val_accuracy: 0.6996
    Epoch 4/50
    50/50 [==============================] - 0s 7ms/step - loss: 0.5981 - accuracy: 0.6902 - val_loss: 0.5348 - val_accuracy: 0.7643
    Epoch 5/50
    50/50 [==============================] - 0s 8ms/step - loss: 0.5814 - accuracy: 0.6952 - val_loss: 0.5215 - val_accuracy: 0.7586
    Epoch 6/50
    50/50 [==============================] - 0s 7ms/step - loss: 0.5875 - accuracy: 0.7079 - val_loss: 0.5359 - val_accuracy: 0.7510
    Epoch 7/50
    50/50 [==============================] - 0s 8ms/step - loss: 0.5721 - accuracy: 0.7130 - val_loss: 0.5277 - val_accuracy: 0.7624
    Epoch 8/50
    50/50 [==============================] - 0s 7ms/step - loss: 0.5721 - accuracy: 0.7067 - val_loss: 0.5178 - val_accuracy: 0.7681
    Epoch 9/50
    50/50 [==============================] - 0s 7ms/step - loss: 0.5685 - accuracy: 0.7168 - val_loss: 0.5195 - val_accuracy: 0.7700
    Epoch 10/50
    50/50 [==============================] - 0s 7ms/step - loss: 0.5573 - accuracy: 0.7187 - val_loss: 0.5153 - val_accuracy: 0.7700
    Epoch 11/50
    50/50 [==============================] - 0s 7ms/step - loss: 0.5385 - accuracy: 0.7429 - val_loss: 0.5108 - val_accuracy: 0.7738
    Epoch 12/50
    50/50 [==============================] - 0s 7ms/step - loss: 0.5352 - accuracy: 0.7384 - val_loss: 0.5133 - val_accuracy: 0.7700
    Epoch 13/50
    50/50 [==============================] - 0s 7ms/step - loss: 0.5305 - accuracy: 0.7613 - val_loss: 0.4998 - val_accuracy: 0.7681
    Epoch 14/50
    50/50 [==============================] - 0s 7ms/step - loss: 0.5362 - accuracy: 0.7371 - val_loss: 0.5041 - val_accuracy: 0.7776
    Epoch 15/50
    50/50 [==============================] - 0s 7ms/step - loss: 0.5301 - accuracy: 0.7454 - val_loss: 0.4948 - val_accuracy: 0.7795
    Epoch 16/50
    50/50 [==============================] - 0s 7ms/step - loss: 0.5179 - accuracy: 0.7549 - val_loss: 0.4999 - val_accuracy: 0.7928
    Epoch 17/50
    50/50 [==============================] - 0s 7ms/step - loss: 0.5247 - accuracy: 0.7505 - val_loss: 0.4913 - val_accuracy: 0.7795
    Epoch 18/50
    50/50 [==============================] - 0s 7ms/step - loss: 0.5123 - accuracy: 0.7530 - val_loss: 0.4837 - val_accuracy: 0.7738
    Epoch 19/50
    50/50 [==============================] - 0s 7ms/step - loss: 0.5074 - accuracy: 0.7587 - val_loss: 0.4936 - val_accuracy: 0.7928
    Epoch 20/50
    50/50 [==============================] - 0s 7ms/step - loss: 0.5051 - accuracy: 0.7638 - val_loss: 0.4838 - val_accuracy: 0.7890
    Epoch 21/50
    50/50 [==============================] - 0s 7ms/step - loss: 0.4838 - accuracy: 0.7822 - val_loss: 0.4797 - val_accuracy: 0.7833
    Epoch 22/50
    50/50 [==============================] - 0s 7ms/step - loss: 0.4927 - accuracy: 0.7644 - val_loss: 0.4762 - val_accuracy: 0.7871
    Epoch 23/50
    50/50 [==============================] - 0s 7ms/step - loss: 0.4893 - accuracy: 0.7714 - val_loss: 0.4777 - val_accuracy: 0.7890
    Epoch 24/50
    50/50 [==============================] - 0s 7ms/step - loss: 0.4911 - accuracy: 0.7790 - val_loss: 0.4736 - val_accuracy: 0.7947
    Epoch 25/50
    50/50 [==============================] - 0s 7ms/step - loss: 0.4717 - accuracy: 0.7892 - val_loss: 0.4719 - val_accuracy: 0.7909
    Epoch 26/50
    50/50 [==============================] - 0s 7ms/step - loss: 0.4769 - accuracy: 0.7740 - val_loss: 0.4697 - val_accuracy: 0.7947
    Epoch 27/50
    50/50 [==============================] - 0s 7ms/step - loss: 0.4737 - accuracy: 0.7898 - val_loss: 0.4725 - val_accuracy: 0.7909
    Epoch 28/50
    50/50 [==============================] - 0s 7ms/step - loss: 0.4718 - accuracy: 0.7829 - val_loss: 0.4868 - val_accuracy: 0.7871
    Epoch 29/50
    50/50 [==============================] - 0s 7ms/step - loss: 0.4696 - accuracy: 0.7822 - val_loss: 0.4720 - val_accuracy: 0.7928
    Epoch 30/50
    50/50 [==============================] - 0s 7ms/step - loss: 0.4648 - accuracy: 0.7829 - val_loss: 0.4690 - val_accuracy: 0.7909
    Epoch 31/50
    50/50 [==============================] - 0s 7ms/step - loss: 0.4632 - accuracy: 0.7854 - val_loss: 0.4757 - val_accuracy: 0.7947
    Epoch 32/50
    50/50 [==============================] - 0s 7ms/step - loss: 0.4689 - accuracy: 0.7854 - val_loss: 0.4712 - val_accuracy: 0.7928
    Epoch 33/50
    50/50 [==============================] - 0s 7ms/step - loss: 0.4624 - accuracy: 0.7835 - val_loss: 0.4626 - val_accuracy: 0.8061
    Epoch 34/50
    50/50 [==============================] - 0s 7ms/step - loss: 0.4500 - accuracy: 0.7937 - val_loss: 0.4670 - val_accuracy: 0.8061
    Epoch 35/50
    50/50 [==============================] - 0s 7ms/step - loss: 0.4550 - accuracy: 0.7873 - val_loss: 0.4656 - val_accuracy: 0.8061
    Epoch 36/50
    50/50 [==============================] - 0s 7ms/step - loss: 0.4462 - accuracy: 0.7956 - val_loss: 0.4596 - val_accuracy: 0.8080
    Epoch 37/50
    50/50 [==============================] - 0s 7ms/step - loss: 0.4534 - accuracy: 0.7956 - val_loss: 0.4705 - val_accuracy: 0.7966
    Epoch 38/50
    50/50 [==============================] - 0s 7ms/step - loss: 0.4310 - accuracy: 0.8089 - val_loss: 0.4583 - val_accuracy: 0.8156
    Epoch 39/50
    50/50 [==============================] - 0s 7ms/step - loss: 0.4392 - accuracy: 0.8133 - val_loss: 0.4636 - val_accuracy: 0.8099
    Epoch 40/50
    50/50 [==============================] - 0s 7ms/step - loss: 0.4394 - accuracy: 0.8076 - val_loss: 0.4653 - val_accuracy: 0.8099
    Epoch 41/50
    50/50 [==============================] - 0s 7ms/step - loss: 0.4476 - accuracy: 0.8159 - val_loss: 0.4631 - val_accuracy: 0.8099
    Epoch 42/50
    50/50 [==============================] - 0s 7ms/step - loss: 0.4445 - accuracy: 0.8057 - val_loss: 0.4573 - val_accuracy: 0.8118
    Epoch 43/50
    50/50 [==============================] - 0s 7ms/step - loss: 0.4390 - accuracy: 0.8210 - val_loss: 0.4580 - val_accuracy: 0.8061
    Epoch 44/50
    50/50 [==============================] - 0s 7ms/step - loss: 0.4605 - accuracy: 0.7879 - val_loss: 0.4565 - val_accuracy: 0.8137
    Epoch 45/50
    50/50 [==============================] - 0s 7ms/step - loss: 0.4306 - accuracy: 0.8089 - val_loss: 0.4627 - val_accuracy: 0.8099
    Epoch 46/50
    50/50 [==============================] - 0s 7ms/step - loss: 0.4416 - accuracy: 0.8038 - val_loss: 0.4683 - val_accuracy: 0.8061
    Epoch 47/50
    50/50 [==============================] - 0s 7ms/step - loss: 0.4374 - accuracy: 0.8057 - val_loss: 0.4647 - val_accuracy: 0.8118
    Epoch 48/50
    50/50 [==============================] - 0s 7ms/step - loss: 0.4227 - accuracy: 0.8197 - val_loss: 0.4644 - val_accuracy: 0.8099
    Epoch 49/50
    50/50 [==============================] - 0s 7ms/step - loss: 0.4077 - accuracy: 0.8267 - val_loss: 0.4804 - val_accuracy: 0.8042
    Epoch 50/50
    50/50 [==============================] - 0s 7ms/step - loss: 0.4276 - accuracy: 0.8159 - val_loss: 0.4603 - val_accuracy: 0.8156



```python
pd.DataFrame(results.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
```


    
![png](output_48_0.png)
    



```python
#evalute the model
test_loss, test_acc = model.evaluate(x_test_scaled, Y_test)
print('Test accuracy:', test_acc)
```

    29/29 [==============================] - 0s 5ms/step - loss: 0.4471 - accuracy: 0.8069
    Test accuracy: 0.8068812489509583



```python
# making prediction
predictions = model.predict(X_test)
```

    29/29 [==============================] - 0s 1ms/step



```python
predicted = tf.squeeze(predictions)
predicted = np.array([1 if x >= 0.5 else 0 for x in predicted])
actual = np.array(Y_test)
conf_mat = confusion_matrix(actual, predicted)
displ = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
displ.plot()
```




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x49b9a3820>




    
![png](output_51_1.png)
    


# Explainability 

## Explain tensorflow keras model with shap DeepExplainer


```python
sample_size=500
if sample_size>len(X_train.values):
  sample_size=len(X_train.values)

explainer = shap.DeepExplainer(model, X_train.values[:sample_size], y_train)
```

    Your TensorFlow version is newer than 2.4.0 and so graph support has been removed in eager mode and some static graphs may not be supported. See PR #1483 for discussion.



```python
# Explain the SHAP values for the whole dataset
# to test the SHAP explainer.
shap_values = explainer.shap_values(X_train.values[:sample_size], ranked_outputs=None)
```

    `tf.keras.backend.set_learning_phase` is deprecated and will be removed after 2020-10-11. To update it, simply pass a True/False value to the `training` argument of the `__call__` method of your layer or model.
    2024-02-20 12:35:40.755383: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type GPU is enabled.



```python
shap_values
```




    [array([[-1.28761018e-02,  2.61684251e-03,  1.22101256e-03, ...,
              8.43746180e-04, -1.53695641e-03,  6.51940797e-03],
            [ 7.81806372e-03,  4.66133794e-03, -3.80936824e-02, ...,
             -3.02560464e-03, -3.34495748e-03,  2.56811804e-03],
            [ 4.50713374e-03, -5.79754077e-03,  1.47271063e-02, ...,
              3.39945336e-03,  4.03275434e-03, -4.77008102e-03],
            ...,
            [-3.10847047e-03,  3.97476554e-03, -1.40036177e-02, ...,
              1.44137791e-03,  7.17193168e-03,  2.29408732e-03],
            [ 1.69399404e-03,  1.97411259e-03,  1.24843428e-02, ...,
             -1.57582830e-03, -2.14419863e-03,  7.04625854e-05],
            [-7.05904374e-03,  2.57816073e-03, -1.22835450e-02, ...,
             -3.26953549e-03, -1.49948173e-03,  3.01212166e-03]])]




```python
# #shap_values 
# shap.initjs()
# shap.force_plot(explainer.expected_value[0].numpy(), shap_values[0], X_train, link="logit")
```


```python
def get_fetaure_importnace(shap_values, features):
    "Features is the  X_train.columns names"
    vals = np.abs(pd.DataFrame(shap_values[0]).values).mean(0)
    shap_importance = pd.DataFrame(list(zip(features, vals)), columns=['col_name', 'feature_importance_vals'])
    shap_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
    shap_importance.loc[shap_importance['feature_importance_vals']>0]
    return shap_importance
```


```python
shap_importance = get_fetaure_importnace(shap_values, features=X_train.columns)
shap_importance
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col_name</th>
      <th>feature_importance_vals</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>131</th>
      <td>ESR1</td>
      <td>0.084175</td>
    </tr>
    <tr>
      <th>245</th>
      <td>BEX1</td>
      <td>0.031682</td>
    </tr>
    <tr>
      <th>279</th>
      <td>FOSB</td>
      <td>0.029250</td>
    </tr>
    <tr>
      <th>110</th>
      <td>IGJ</td>
      <td>0.029118</td>
    </tr>
    <tr>
      <th>271</th>
      <td>FABP7</td>
      <td>0.027046</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TAT</td>
      <td>0.000288</td>
    </tr>
    <tr>
      <th>241</th>
      <td>CHGB</td>
      <td>0.000264</td>
    </tr>
    <tr>
      <th>152</th>
      <td>FOLR1</td>
      <td>0.000237</td>
    </tr>
    <tr>
      <th>198</th>
      <td>PVALB</td>
      <td>0.000217</td>
    </tr>
    <tr>
      <th>72</th>
      <td>S100P</td>
      <td>0.000185</td>
    </tr>
  </tbody>
</table>
<p>295 rows × 2 columns</p>
</div>




```python
def plot_importance_features(Features, top=10, title=" "):
    " Features is a 2 colunm dataframe with features names and weights"
    
    fig, ax = plt.subplots(figsize =(5, 5))
    top_features = Features.iloc[:top]
    # Horizontal Bar Plot
    ax.barh(top_features['col_name'], top_features['feature_importance_vals'])
    ax.set_yticks(top_features['col_name'])
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Importance')
    ax.set_title(title)
    plot = plt.show()
    return plot
```


```python
plot_importance_features(Features=shap_importance, top=10, title="Shap importance features")
```


    
![png](output_61_0.png)
    



```python
# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
shap.summary_plot(shap_values[0], X_test, plot_type="bar", max_display=10)
```


    
![png](output_62_0.png)
    



```python
shap.summary_plot(shap_values[0], features=X_train.values[:sample_size], feature_names=features, show=False, plot_type="dot", max_display=10)
```


    
![png](output_63_0.png)
    


## Explain tensorflow keras model with weights


```python
def get_weights_importnace(model, features, num=0):
    "Features is the  X_train.columns names"
    vals_weights = np.abs(model.layers[num].get_weights()[num].T).mean(num)
    weights_importance = pd.DataFrame(list(zip(features, vals_weights)), columns=['col_name', 'feature_importance_vals'])
    weights_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
    weights_importance.loc[weights_importance['feature_importance_vals']>0] 
    return weights_importance
```


```python
weights_importance =  get_weights_importnace(model, features= X_train.columns)
weights_importance
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col_name</th>
      <th>feature_importance_vals</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>131</th>
      <td>ESR1</td>
      <td>0.190406</td>
    </tr>
    <tr>
      <th>16</th>
      <td>AGTR1</td>
      <td>0.136717</td>
    </tr>
    <tr>
      <th>182</th>
      <td>DCD</td>
      <td>0.136054</td>
    </tr>
    <tr>
      <th>121</th>
      <td>KLK5</td>
      <td>0.135646</td>
    </tr>
    <tr>
      <th>110</th>
      <td>IGJ</td>
      <td>0.134122</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>212</th>
      <td>PRODH</td>
      <td>0.045159</td>
    </tr>
    <tr>
      <th>194</th>
      <td>AKR1C2</td>
      <td>0.044727</td>
    </tr>
    <tr>
      <th>176</th>
      <td>ACTG2</td>
      <td>0.044115</td>
    </tr>
    <tr>
      <th>91</th>
      <td>SLPI</td>
      <td>0.042295</td>
    </tr>
    <tr>
      <th>198</th>
      <td>PVALB</td>
      <td>0.033221</td>
    </tr>
  </tbody>
</table>
<p>295 rows × 2 columns</p>
</div>




```python
plot_importance_features(Features=weights_importance, top=10, title="Weights importance features")
```


    
![png](output_67_0.png)
    


# Hyperparameter tunning with GridSearchCV using scikeras.wrappers


```python

def get_model(nb_hidden, nb_neurons):
    print(f"##  {nb_hidden}-{nb_neurons}   ##")
    model = keras.models.Sequential()
    model.add(Dense(64, input_shape=x_train_scaled.shape[1:], activation = 'relu', name='dense_layer'))
    model.add(keras.layers.Dense(2 * nb_neurons, activation="relu"))    
    if nb_hidden > 1 :
        for layer in range(2, nb_hidden):
            model.add(keras.layers.Dense(nb_neurons, activation="relu"))
    model.add(keras.layers.Dense(1, activation="sigmoid" ))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model

```


```python
#!pip install scikeras
from scikeras.wrappers import KerasClassifier 
```


```python
keras_classifier = KerasClassifier(get_model, nb_hidden=1, nb_neurons=1)


param_distribs = {
"nb_hidden": [1,2,3,6,8],
"nb_neurons": [3,5,8,10],
}



early_stopping = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

search_cv = GridSearchCV(keras_classifier, param_distribs,  cv=2 )
search_cv.fit(x_train_scaled, y_train, epochs=20,  callbacks=[early_stopping], validation_data=[x_val_scaled, y_val],  )

print(f"search_cv.best_params_ = {search_cv.best_params_}")

print(f"search_cv.best_score_ = {search_cv.best_score_}") 
```


```python
print(f"search_cv.best_params_ = {search_cv.best_params_}")

print(f"search_cv.best_score_ = {search_cv.best_score_}") 
```

    search_cv.best_params_ = {'nb_hidden': 1, 'nb_neurons': 10}
    search_cv.best_score_ = 0.810161475499713


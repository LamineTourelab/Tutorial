```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

# Data load and preprocessing 

## Expression data


```python
df = pd.read_table("~/Downloads/GLORY_20170922.genes_all_samples_TPM.txt", sep="\t", index_col=0)
df
```


```python
df.index = df['hugo_id']
df = df.drop(['hugo_id'], axis=1)
df = df.T
df.head()
```


```python
df.info()
```


```python
duplicate_columns = df.columns[df.columns.duplicated()]
duplicate_columns
```


```python
df = df.drop(columns=duplicate_columns)
df.head()
```

## label data 


```python
samples = pd.read_excel('~/Downloads/GBM_GLORY_sample_info.xlsx', index_col=0)
samples.head()
```


```python
Outcome = pd.DataFrame(samples['IDH1_STATUS-sequencing'])
Outcome.head()
```


```python
Outcome = Outcome.replace("MUT",1)
Outcome = Outcome.replace("WT",0)
Outcome.head()
```


```python
Outcome = Outcome[Outcome.index.isin(df.index)]
Outcome.head()
```


```python
print('Labels counts in Outcome:', np.bincount(Outcome['IDH1_STATUS-sequencing']))
```

here we clearly dealing with class imbalance.
### Class imbalance correct using imblearn


```python
#!pip install -U imbalanced-learn
```


```python
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
df_resampled, Outcome_resampled = ros.fit_resample(df, Outcome['IDH1_STATUS-sequencing'])
```


```python
from collections import Counter
print(sorted(Counter(Outcome_resampled).items()))
```


```python
print('Labels counts in Outcome:', np.bincount(Outcome_resampled))
```

# Explainability

## Lime


```python
#!pip install lime
import lime 
```


```python
X_train, X_test, y_train, y_test = train_test_split(df_resampled, Outcome_resampled, test_size=0.3, random_state=42)

model = xgb.XGBClassifier(objective ='binary:logistic', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
model.fit(X_train, y_train)
pred= model.predict(X_test)
```


```python
y_pred = pd.DataFrame(model.predict(X_test), columns=['pred'], index=X_test.index)
y_pred.head()
```


```python
explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns,
    class_names= ['WT', 'MUT'],
    mode='classification'
)
```


```python
random.seed(10)
# Choose the instance and use it to predict the results. Here I use the 30th (below the 334 patient). 
exp = explainer.explain_instance(
    data_row=X_test.iloc[8], 
    #num_features: maximum number of features present in explanation. I keept default 10.
    #num_samples: size of the neighborhood to learn the linear model.Default 500.
    predict_fn=model.predict_proba
)
exp.show_in_notebook(show_table=True)
```


```python
# Show the results as list.
exp.as_list()
```


```python
%matplotlib inline
fig = exp.as_pyplot_figure()
```


```python
from lime import submodular_pick
```


```python
# Let's use SP-LIME to return explanations on a few sample data sets 
# and obtain a non-redundant global decision perspective of the black-box model
sp_exp = submodular_pick.SubmodularPick(explainer, 
                                        np.array(X_test),
                                        model.predict_proba,
                                        num_features=10,
                                        num_exps_desired=2)   
```


```python
[exp.as_pyplot_figure(label=exp.available_labels()[0]) for exp in sp_exp.sp_explanations]
print('SP-LIME Local Explanations')
```

## Shap


```python
#!pip install shap
import shap
```


```python
explainer=shap.Explainer(model)
shap_values = explainer(X_train)
shap_contrib = explainer.shap_values(X_test)
```


```python
#import Javascript

#shap_values
shap.initjs(),
shap.plots.force(shap_values)
```


```python
#shap.plots.waterfall(shap_values[0])
```


```python
# Global bar plot
shap.plots.bar(shap_values, max_display=8)
```


```python
# Local bar plot for the patient 334 (index 8).
shap.plots.bar(shap_values[8], max_display=8)
```


```python
shap.plots.beeswarm(shap_values, max_display=8)
```

# Explainability Tools

## Shapash


```python
response_dict = {0: 'WT', 1:'MUT'}
```


```python
#!pip install shapash
```


```python

#
from shapash import SmartExplainer
import tkinter as TK


xpl = SmartExplainer(
    model=model,
    #backend='shap' is the default.
    label_dict=response_dict  #Dictionary mapping integer labels to domain names (classification - target values).
    #preprocessing=encoder,   # Optional: compile step can use inverse_transform method
    #features_dict=name # optional parameter, specifies label for features name 
)

xpl.compile(
   contributions=shap_contrib, # Shap Contributions pd.DataFrame
    y_pred=y_pred,  #Prediction values (1 column only)
    x=X_test,  # a preprocessed dataset: Shapash can apply the model to it
    y_target=y_test #Target values (1 column only). 
)

```


```python
xpl.plot.features_importance(max_features=10, label=1)
xpl.plot.scatter_plot_prediction()
#y_pred
xpl.filter(max_contrib=10)
xpl.plot.local_plot(index=334, label='MUT')
xpl.plot.compare_plot(row_num=[0, 1, 2, 3, 4, 5, 6], max_features=8)

```


```python
#Start WebApp
app = xpl.run_app(port=8850, title_story='Explanation')
# Kill the wepapp
app.kill()
```

## Explainerdashboard


```python
#!pip install explainerdashboard
```


```python
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
```


```python
patient_idx=X_test.index
patient_idx
```


```python
explainerdb = ClassifierExplainer(model, X_test, y_test, 
                                    X_background=X_train, 
                                    model_output='y_pred',
                                    idxs=patient_idx,
                                    labels=['WT', 'MUT'])
```


```python
db = ExplainerDashboard(explainerdb)
```


```python
#Run the dashboard webApp
db.run(port=8050, mode='external')
```


```python
db.terminate(8050)
```


```python

```


```python

```

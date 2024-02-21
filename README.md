# Tutorials

## Tutorials on machine learning in general and in bioinformatics in biomedical research
In this repository, you will find tutorials about AI in general and specifically in medical research. 

The repository is divided into differents sections where each section contains tutorials about an erea in data science field like NLP, Image processing and analysis, graph etc. Fill free to propose update or to collaborate. I tried to do all the code in colab to make it easy for reproducibility. Some code are not mine!.

This tutorial is available on my github page:

[https://laminetourelab.github.io/](https://laminetourelab.github.io/)

```
conda create -n mkdoc python=3.11  
pip install -U mkdocs
pip install mkdocs-jupyter
pip install python-markdown-math
pip install markdown-callouts
pip install mdx-gh-links
pip install mkdocs-click

mkdocs gh-deploy --config-file mkdocs.yml --remote-branch main
```

# User Guide Index

## `Bioinformatics`

- [RNA Velocity analysis scVelo](https://github.com/LamineTourelab/Tutorial/blob/main/Bioinformatics/SingleCellData_Tutorial/RNA_Velocity_analysis_scVelo.ipynb)
- [SingleCell Data Preprocessing with scanpy](https://github.com/LamineTourelab/Tutorial/blob/main/Bioinformatics/SingleCellData_Tutorial/SingleCellData_Preprocessing_with_scanpy.ipynb)
- [Trajectory inference With CellRank](https://github.com/LamineTourelab/Tutorial/blob/main/Bioinformatics/SingleCellData_Tutorial/Trajectory_inference_With_CellRank.ipynb)

## `Machine learning`
- [Introduction to Autoencoder](https://github.com/LamineTourelab/Tutorial/blob/main/machine%20learning/Autoencoder.ipynb)
- [Decision Trees using iris dataset](https://github.com/LamineTourelab/Tutorial/blob/main/machine%20learning/DecisionTrees_using_iris_dataset.ipynb)
- [Transfer learning using VGG16](https://github.com/LamineTourelab/Tutorial/blob/main/machine%20learning/Transfer_learning_using_VGG16.ipynb)
- [Introductio to tensorflow API tutorial](https://github.com/LamineTourelab/Tutorial/blob/main/machine%20learning/tensorflow_API_tutorial.ipynb)
- [Introductio to tensor basics operation](https://github.com/LamineTourelab/Tutorial/blob/main/machine%20learning/tensor_basics_operation.ipynb)

## `Explainable AI`

- [Explainable xgboot model using shap and lime](https://github.com/LamineTourelab/Tutorial/blob/main/Explainable%20AI/explainability_shap%26lime.ipynb)
- [Explain tensorflow model using Shap and Model Weights ](https://github.com/LamineTourelab/Tutorial/blob/main/Explainable%20AI/Explainable_tensorflow_model_Shap.ipynb)
- [Explainable tensorflow multiclass model using Shap and model weights](https://github.com/LamineTourelab/Tutorial/blob/main/Explainable%20AI/Explainable_tensorflow_multiclass_model_using_Shap_and_model_weights.ipynb)

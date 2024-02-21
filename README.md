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

- [RNA_Velocity_analysis_scVelo](user-guide/Bioinformatics/SingleCellData/RNA_Velocity_analysis_scVelo.md)
- [SingleCellData_Preprocessing_with_scanpy](user-guide/Bioinformatics/SingleCellData/SingleCellData_Preprocessing_with_scanpy.md)
- [Trajectory_inference_With_CellRank](user-guide/Bioinformatics/SingleCellData/Trajectory_inference_With_CellRank/Trajectory_inference_With_CellRank.md)
- [Readme](user-guide/Bioinformatics/SingleCellData/Readme.md)

## `Machine learning`
- [Autoencoder](user-guide/Machine learning/Autoencoder/Autoencoder.md)
- [DecisionTrees_using_iris_dataset](user-guide/Machine learning/DecisionTrees_using_iris_dataset/DecisionTrees_using_iris_dataset.md)
- [Transfer_learning_using_VGG16](user-guide/Machine learning/Transfer_learning_using_VGG16/Transfer_learning_using_VGG16.md)
- [tensorflow_API_tutorial](user-guide/Machine learning/tensorflow_API_tutorial/tensorflow_API_tutorial.md)

## `Explainable AI`

- [explainability_shap&lime](user-guide/Explainable AI/explainability_shap&lime/explainability_shap&lime.md)
- [Explainable_tensorflow_model_Shap](user-guide/Explainable AI/Explainable_tensorflow_model_Shap/Explainable_tensorflow_model_Shap.md)

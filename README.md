# `Tutorials`

## Tutorials on explainable machine learning, artifical intelligence in general and in bioinformatics in biomedical research
In this repository, you will find tutorials about artifical intelligence applied in medical research. 

The repository is divided into differents sections where each section contains tutorials about an erea in data science field like NLP, Image processing and analysis, graph etc. Fill free to propose update or to collaborate. I tried to do all the code in colab to make it easy for reproducibility. Some code are not mine!.

# `Links`

- Documentation: https://laminetourelab.github.io
- Contributing: https://github.com/LamineTourelab/Tutorial/graphs/contributors

The Mkdocs was used for the creation of the website.

# `Publish your Markdown docs on GitHub Pages`


First install mkdocs tools with pip:
```
conda create -n mkdoc python=3.11
conda activate mkdoc
pip install -U mkdocs
pip install mkdocs-jupyter
pip install python-markdown-math
pip install markdown-callouts
pip install mdx-gh-links
pip install mkdocs-click
```
To create your docs source project, you will need the repository source cloned:

```mkdocs new GITHUB_repo```
Now if you go at this repo you will fine these files:

```
./
-- docs/
-- mkdocs.yml
```
Run ```mkdocs serve``` to test your browser.

The deployment can be done inside your github.io repository as follow:

```mkdocs gh-deploy --config-file ../GITHUB_repo/mkdocs.yml --remote-branch main```

For more details visit the [mkdocs.org](https://www.mkdocs.org/) website.

# `User Guide Index`

## `Bioinformatics`

- [RNA Velocity analysis scVelo](https://github.com/LamineTourelab/Tutorial/blob/main/Bioinformatics/SingleCellData_Tutorial/RNA_Velocity_analysis_scVelo.ipynb)
- [SingleCell Data Preprocessing with scanpy](https://github.com/LamineTourelab/Tutorial/blob/main/Bioinformatics/SingleCellData_Tutorial/SingleCellData_Preprocessing_with_scanpy.ipynb)
- [Trajectory inference With CellRank](https://github.com/LamineTourelab/Tutorial/blob/main/Bioinformatics/SingleCellData_Tutorial/Trajectory_inference_With_CellRank.ipynb)

## ` Medical Image Analysis`
- [Chest X Ray Images classification with DenseNet121 and explain with GradCam on tensorflow 2](https://github.com/LamineTourelab/Tutorial/blob/main/Images/Xray_classification_with_densenet121_and_gradcam.ipynb)
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
- [Chest X Ray Images classification with DenseNet121 and explain with GradCam on tensorflow 2](https://github.com/LamineTourelab/Tutorial/blob/main/Explainable%20AI/Xray_classification_with_densenet121_and_gradcam.ipynb)

  ### Note 
  There is an ongoing project in explainable AI applied to healthcare in collaboration with [Salvatore RAIELI](https://www.linkedin.com/in/salvatore-raieli/). The source code can be found at [https://github.com/SalvatoreRa/explanaibleAI](https://github.com/SalvatoreRa/explanaibleAI). Don't hesitate to reach out for collaboration or any question.

## `Data visualization`
- [Exploratory data analysis using matplotlib plotly express and Dash](https://github.com/LamineTourelab/Tutorial/blob/main/DataViz/EDA_matplotlib_Dashboard_Dataviz.ipynb)
- [Shiny App in R](https://github.com/LamineTourelab/Tutorial/blob/main/DataViz/app.R)

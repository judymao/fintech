# FinTech
ECO416/FIN516 Final Project

# Scripts
## Data
- To run EDA scripts, please download the dataset from https://www.kaggle.com/datasets/ealaxi/paysim1 and name it "PS_data.csv". This should be placed in the same directory as the GitHub scripts. It is not included here due to GitHub's file size limitations.
- Files "PS_processed_data_batch1.csv" through "PS_processed_data_batch19.csv" are batched datasets that include our engineered features for model development. These files are used to generate training and testing datasets.

## EDA
- [EDA.ipynb](EDA.ipynb) and [Time Series EDA.ipynb](Time%20Series%20EDA.ipynb) - Generate plots for exploratory data analysis

## Models
Scripts to perform cross-validation for model fitting process and report testing metrics for optimized models
- [model_logit.ipynb](model_logit.ipynb) - Logistic Regression
- [model_nn.ipynb](model_nn.ipynb) - Neural Network
- [model_rf.ipynb](model_rf.ipynb) - Decision Tree, Decision Tree with Boosting, and Random Forest
- [model_svm.ipynb](model_svm.ipynb) - SVM

## Other
- [utils.py](utils.py) - Utility script to standardize certain functionalities, such as train/test data splitting process and model comparison

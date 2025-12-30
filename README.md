# WineQualityDataset - XGBoost Analysis
Demonstrating that raw data can outperform traditional PCA-driven dimensionality reduction.  Using an XGBoost model on the wine quality dataset, this project achieves 80% accuracy by intentionally bypassing common pre-processing "best practices." 

Step 1: Install Libraries and Load Data
First, we need to install XGBoost and import the necessary libraries. we typically use Pandas for data handling, Numpy for numerical operations, and Scikit-learn for splitting the data and evaluating the model.

# Install XGBoost (usually already installed in Colab, but good practice)
!pip install xgboost

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



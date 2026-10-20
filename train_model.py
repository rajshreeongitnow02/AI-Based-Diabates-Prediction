import pandas as pd
import nupy as np
import joblib

from sklearn.model_selection import trains_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForsetClassifier 
from sklearn.metrics import(accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,classification_report)

#load data


import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import(accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,classification_report)

data_path = "Dataset/diabetes_binary_health (1).csv"
df= pd.read_csv(data_path)
print("Dataset loaded successfully!")
print("Shape: ", df.shape)
print("\nFirst 5 rows:")
print(df.head())

x=df.drop("Diabetes", axis=1)#separating values
y=df["Diabetes"]
print("\nFeatures:")
print(x.columns.tolist())
imputer=SimpleImputer(strategy="median")
x= pd.DataFrame(
    imputer.fit_transform(x),
    columns=x.columns
)
print("\nMissing values handled successfully.")
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.20,random_state=42,stratify=y)
print("\nTraining samples:", len(xtrain))
print("Testing samples:", len(xtest))

#random forest review
model = RandomForestClassifier(n_estimators=200,random_state=42,class_weight="balanced")
model.fit(xtrain,ytrain)
print("\nRandom Forest model trained successfully!")
ypred= model.predict(xtest)
yprob= model.predict_proba(xtest)[:, 1]
acc= accuracy_score(ytest, ypred)
preci=precision_score(ytest, ypred,zero_division=0)
recall=  recall_score(ytest,ypred,zero_division=0)
f1=f1_score(ytest,ypred,zero_division=0)
rocauc = roc_auc_score(ytest,yprob)

print("\n MODEL EVALUATION ")

print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {preci:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"ROC-AUC   : {rocauc:.4f}")
print("\nClassification Report:")
print(classification_report(ytest, ypred,zero_division=0))

importance = pd.DataFrame({
    "Feature": x.columns,
    "Importance": model.feature_importances_})
importance = importance.sort_values(by="Importance",ascending=False)
print("\n FEATURE IMPORTANCE ")
print(importance)
joblib.dump(
    model,
    "model/diabetes_random_forest.pkl"
)
joblib.dump(
    imputer,
    "model/diabetes_imputer.pkl")
print("\nModel saved successfully!")
print("Files created inside the model folder.")

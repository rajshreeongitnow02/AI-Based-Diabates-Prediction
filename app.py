import pandas as pd
import joblib

model= joblib.load("Model/diabetes_random_forest.pkl")
imputer= joblib.load("Model/diabetes_imputer.pkl")
print("=" * 50)
print("AI-BASED DIABETES PREDICTION SYSTEM")
print("=" * 50)
print("\nEnter the patient's health information.\n")

#taking inputs from user about their data
age= float(input("Enter Age: "))
bmi= float(input("Enter BMI: "))
physical_activity = int(input("Physical Activity (0 = Low/No, 1 = Active): "))
blood_pressure = float(input("Enter Blood Pressure: "))
cholesterol = int(input("Cholesterol (0 = Normal, 1 = High): "))
glucose = float(input("Enter Glucose Level: "))

input_data = pd.DataFrame({
    "Age": [age],
    "BMI": [bmi],
    "PhysicalActivity": [physical_activity],
    "BloodPressure": [blood_pressure],
    "Cholesterol": [cholesterol],
    "Glucose": [glucose]
})

input_data = pd.DataFrame(imputer.transform(input_data),columns=input_data.columns)
prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0][1]
risk_percentage = probability * 100

print("\n" + "=" * 50)
print("RESULT")
print("=" * 50)

if prediction == 1:
    print("\nPrediction: DIABETIC!")
else:
    print("\nPrediction: NON-DIABETIC  ^_^ ")

print(f"Estimated Diabetes Probability: {risk_percentage:.2f}%")
print("\n" + "=" * 50)
print("\nNote: This is an educational machine-learning project not a medical diagnosis")

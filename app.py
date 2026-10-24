from flask import Flask, request, render_template_string
import pandas as pd
import joblib

app = Flask(__name__)
# load trained model
model = joblib.load("diabetes_random_forest.pkl")
imputer = joblib.load("diabetes_imputer.pkl")

# web app page
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Diabetes Prediction System</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 700px;
            margin: 40px auto;
            padding: 20px;
        }
        h1 {
            text-align: center;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }

        input, select {
            width: 100%;
            padding: 10px;
            box-sizing: border-box;
        }
        button {
            width: 100%;
            padding: 12px;
            margin-top: 15px;
            cursor: pointer;
        }
        .result {
            margin-top: 25px;
            padding: 20px;
            text-align: center;
            border: 1px solid #ccc;
        }
        .note { margin-top: 20px;font-size: 13px; }
    </style>
</head>

<body>
<h1>AI-Based Diabetes Prediction System</h1>
<p>
Enter the patient's health information to predict
the likelihood of diabetes.
</p>
<form method="POST">
    <div class="form-group">
        <label>Age</label>
        <input type="number" name="age"min="18" max="100" required>
    </div>

    <div class="form-group">
        <label>BMI</label>
        <input type="number" name="bmi" min="10" max="60" step="0.1" required>
    </div>

    <div class="form-group">
        <label>Physical Activity</label>
        <select name="physical_activity">
            <option value="0">Low / No</option>
            <option value="1">Active</option>
        </select>
    </div>

    <div class="form-group">
        <label>Blood Pressure</label>
        <input type="number" name="blood_pressure" min="70" max="220" required>
    </div>

    <div class="form-group">
        <label>Cholesterol</label>
        <select name="cholesterol">
            <option value="0">Normal</option>
            <option value="1">High</option>
        </select>
    </div>

    <div class="form-group">
        <label>Glucose Level</label>
        <input type="number" name="glucose" min="50" max="300" required>
    </div>
    
    <button type="submit">
        Predict Diabetes
    </button>
</form>

{% if result %}
<div class="result">
    <h2>{{ result }}</h2>
    <p>Estimated Diabetes Probability: <strong>{{ probability }}%</strong> </p>
</div>
{% endif %}
<div class="note">
    <strong>Note:</strong>
    This is an educational machine-learning project
    and is not a medical diagnosis.
</div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    probability = None

    if request.method == "POST":
        age = float(request.form["age"])
        bmi = float(request.form["bmi"])
        physical_activity = int(request.form["physical_activity"])
        blood_pressure = float(request.form["blood_pressure"] )
        cholesterol = int( request.form["cholesterol"] )
        glucose = float(request.form["glucose"]  )
        input_data = pd.DataFrame({
            "Age": [age],
            "BMI": [bmi],
            "PhysicalActivity": [physical_activity],
            "BloodPressure": [blood_pressure],
            "Cholesterol": [cholesterol],
            "Glucose": [glucose]
        })
        input_data = pd.DataFrame(
            imputer.transform(input_data),
            columns=input_data.columns
        )
        prediction = model.predict(input_data)[0]
        probability_value = (model.predict_proba(input_data)[0][1]     )
        probability = f"{probability_value * 100:.2f}"

        if prediction == 1:
            result = "Prediction: DIABETIC"
        else:
            result = "Prediction: NON-DIABETIC"

    return render_template_string(
        HTML,
        result=result,
        probability=probability
    )

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )

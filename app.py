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
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Diabetes Prediction</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            font-family: Arial, Helvetica, sans-serif;
            background: #f4f8fc;
            color: #172033;
            min-height: 100vh;
        }

        /* NAVBAR */

        .navbar {
            width: 100%;
            background: white;
            border-bottom: 1px solid #e5eaf0;
            padding: 18px 7%;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .logo {
            font-size: 21px;
            font-weight: 700;
            color: #1769aa;
        }

        .logo span {
            color: #19a974;
        }

        .nav-tag {
            background: #eef7ff;
            color: #1769aa;
            padding: 7px 14px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
        }

        /* HERO */

        .hero {
            max-width: 1100px;
            margin: 55px auto 30px;
            padding: 0 25px;
            text-align: center;
        }

        .badge {
            display: inline-block;
            background: #e8f7f2;
            color: #14865c;
            padding: 8px 16px;
            border-radius: 30px;
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 18px;
        }

        .hero h1 {
            font-size: 44px;
            line-height: 1.15;
            margin-bottom: 16px;
            color: #14213d;
        }

        .hero h1 span {
            color: #1769aa;
        }

        .hero p {
            max-width: 650px;
            margin: auto;
            color: #667085;
            font-size: 16px;
            line-height: 1.7;
        }

        /* MAIN CARD */

        .main-card {
            max-width: 850px;
            margin: 35px auto 60px;
            background: white;
            border-radius: 18px;
            padding: 35px;
            box-shadow: 0 12px 40px rgba(30, 70, 100, 0.10);
            border: 1px solid #e8edf3;
        }

        .card-header {
            margin-bottom: 30px;
        }

        .card-header h2 {
            font-size: 24px;
            margin-bottom: 7px;
            color: #172033;
        }

        .card-header p {
            color: #7a8494;
            font-size: 14px;
        }

        /* FORM */

        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 22px;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        label {
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 8px;
            color: #344054;
        }

        input,
        select {
            width: 100%;
            padding: 13px 14px;
            border: 1px solid #d7dee8;
            border-radius: 9px;
            background: #fbfcfe;
            color: #172033;
            font-size: 14px;
            outline: none;
            transition: 0.2s;
        }

        input:focus,
        select:focus {
            border-color: #1769aa;
            box-shadow: 0 0 0 3px rgba(23, 105, 170, 0.10);
            background: white;
        }

        .full-width {
            grid-column: 1 / -1;
        }
        button {
            width: 100%;
            border: none;
            padding: 15px;
            border-radius: 10px;
            background: #1769aa;
            color: white;
            font-size: 16px;
            font-weight: 700;
            cursor: pointer;
            transition: 0.2s;
            margin-top: 5px;
        }

        button:hover {
            background: #12558b;
            transform: translateY(-1px);
            box-shadow: 0 7px 18px rgba(23, 105, 170, 0.25);
        }
        .result {
            margin-top: 30px;
            padding: 28px;
            border-radius: 14px;
            text-align: center;
            background: #f5fbff;
            border: 1px solid #d8ecfa;
        }

        .result-label {
            font-size: 13px;
            color: #667085;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }

        .result h2 {
            font-size: 28px;
            color: #1769aa;
            margin-bottom: 10px;
        }

        .probability {
            font-size: 16px;
            color: #475467;
        }

        .probability strong {
            color: #14865c;
            font-size: 20px;
        }

        /* INFO CARDS */

        .info-section {
            max-width: 850px;
            margin: -25px auto 60px;
            padding: 0 25px;
        }

        .info-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
        }

        .info-card {
            background: white;
            padding: 22px;
            border-radius: 12px;
            border: 1px solid #e8edf3;
        }

        .info-card h3 {
            font-size: 15px;
            margin-bottom: 8px;
            color: #1769aa;
        }

        .info-card p {
            font-size: 13px;
            color: #667085;
            line-height: 1.6;
        }

        .note {
            max-width: 850px;
            margin: 0 auto 40px;
            padding: 0 25px;
            text-align: center;
            font-size: 12px;
            color: #7a8494;
            line-height: 1.6;
        }

        footer {
            background: #14213d;
            color: #cbd5e1;
            text-align: center;
            padding: 22px;
            font-size: 12px;
        }

        @media (max-width: 700px) {

            .hero {
                margin-top: 35px;
            }

            .hero h1 {
                font-size: 32px;
            }

            .form-grid {
                grid-template-columns: 1fr;
            }

            .full-width {
                grid-column: auto;
            }

            .info-grid {
                grid-template-columns: 1fr;
            }

            .main-card {
                margin: 25px 15px 50px;
                padding: 25px 20px;
            }
        }
    </style>
</head>
<body>

<nav class="navbar">
    <div class="logo">
        AI<span>Health</span>
    </div>

    <div class="nav-tag">
        Machine Learning Project
    </div>
</nav>

<section class="hero">
    <div class="badge">
        AI-POWERED HEALTH ANALYSIS
    </div>
    <h1>
        Diabetes Risk <span>Prediction</span>
    </h1>
    <p>
        Enter basic health information and our machine-learning
        model will estimate the likelihood of diabetes based on
        the provided health parameters.
    </p>
</section>

<div class="main-card">
    <div class="card-header">
        <h2>Patient Information</h2>
        <p> Enter the required information below to generate a prediction. </p>
    </div>
    <form method="POST">
        <div class="form-grid">
            <div class="form-group">
                <label>Age</label>

                <input
                    type="number"
                    name="age"
                    min="18"
                    max="100"
                    placeholder="e.g. 45"
                    required
                >
            </div>

            <div class="form-group">
                <label>BMI</label>
                <input
                    type="number"
                    name="bmi"
                    min="10"
                    max="60"
                    step="0.1"
                    placeholder="e.g. 25.4"
                    required
                >
            </div>

            <div class="form-group">
                <label>Physical Activity</label>

                <select name="physical_activity">

                    <option value="0">
                        Low / No
                    </option>

                    <option value="1">
                        Active
                    </option>

                </select>
            </div>


            <div class="form-group">
                <label>Blood Pressure</label>

                <input
                    type="number"
                    name="blood_pressure"
                    min="70"
                    max="220"
                    placeholder="e.g. 120"
                    required
                >
            </div>


            <div class="form-group">
                <label>Cholesterol</label>

                <select name="cholesterol">
                    <option value="0">
                        Normal
                    </option>
                    <option value="1">
                        High
                    </option>

                </select>
            </div>

            <div class="form-group">
                <label>Glucose Level</label>

                <input
                    type="number"
                    name="glucose"
                    min="50"
                    max="300"
                    placeholder="e.g. 110"
                    required >
            </div>

            <div class="full-width">

                <button type="submit">
                    Predict Diabetes Risk
                </button>
            </div>
        </div>
    </form>
    {% if result %}
    <div class="result">
        <div class="result-label">
            Prediction Result
        </div>

        <h2>
            {{ result }}
        </h2>

        <div class="probability">
            Estimated Diabetes Probability:
            <strong>{{ probability }}%</strong>
        </div>
    </div>
    {% endif %}
</div>

<section class="info-section">

    <div class="info-grid">

        <div class="info-card">
            <h3>AI Model</h3>
            <p>
                Random Forest classification model trained
                using patient health data.
            </p>
        </div>

        <div class="info-card">
            <h3>Health Parameters</h3>
            <p>
                The model analyses age, BMI, physical activity,
                blood pressure, cholesterol and glucose.
            </p>
        </div>

        <div class="info-card">
            <h3>Prediction</h3>
            <p>
                The system provides an estimated probability
                based on the trained machine-learning model.
            </p>
        </div>
    </div>
</section>
<div class="note">
    <strong>Important:</strong>
    This application is an educational machine-learning project
    and is not intended to provide medical diagnosis or replace
    professional medical advice.
</div>
<footer>
    AI-Based Diabetes Prediction System
    <br>
    Built with Python, Flask & Machine Learning
</footer>
</body>
</html>
"""
```

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
        debug=False
    )

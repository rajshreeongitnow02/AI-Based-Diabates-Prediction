from flask import Flask, request, render_template_string
import pandas as pd
import joblib

app = Flask(__name__)
# load trained model
model = joblib.load("diabetes_random_forest.pkl")
imputer = joblib.load("diabetes_imputer.pkl")

# web app page
```python
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
        }

        body {
            margin: 0;
            font-family: Arial, Helvetica, sans-serif;
            background: #eaf4ff;
            color: #1e293b;
        }

        /* TOP HEADER */

        .header {
            background: #2563eb;
            color: white;
            text-align: center;
            padding: 30px 20px;
            box-shadow: 0 3px 12px rgba(0,0,0,0.12);
        }

        .header h1 {
            margin: 0;
            font-size: 32px;
        }

        .header p {
            margin: 10px 0 0;
            font-size: 15px;
            color: #dbeafe;
        }

        /* MAIN AREA */

        .container {
            width: 90%;
            max-width: 850px;
            margin: 40px auto;
        }

        /* FORM CARD */

        .card {
            background: white;
            padding: 35px;
            border-radius: 15px;
            box-shadow: 0 5px 25px rgba(37, 99, 235, 0.12);
            border: 1px solid #dbeafe;
        }

        .card h2 {
            margin-top: 0;
            color: #1d4ed8;
            font-size: 24px;
        }

        .card-description {
            color: #64748b;
            margin-bottom: 28px;
        }

        /* FORM */

        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        label {
            font-weight: bold;
            margin-bottom: 8px;
            color: #334155;
        }

        input,
        select {
            width: 100%;
            padding: 13px;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            background: #f8fafc;
            font-size: 15px;
            color: #1e293b;
            outline: none;
        }

        input:focus,
        select:focus {
            border-color: #2563eb;
            background: white;
            box-shadow: 0 0 0 3px rgba(37,99,235,0.12);
        }

        /* BUTTON */

        .predict-btn {
            grid-column: 1 / -1;
            padding: 15px;
            margin-top: 5px;
            border: none;
            border-radius: 8px;
            background: #2563eb;
            color: white;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: 0.2s;
        }

        .predict-btn:hover {
            background: #1d4ed8;
            transform: translateY(-1px);
        }

        /* RESULT */

        .result {
            margin-top: 30px;
            padding: 25px;
            text-align: center;
            border-radius: 12px;
            background: #eff6ff;
            border: 2px solid #93c5fd;
        }

        .result h2 {
            margin: 0 0 10px;
            color: #1d4ed8;
            font-size: 25px;
        }

        .result p {
            margin: 0;
            color: #475569;
            font-size: 16px;
        }

        .result strong {
            color: #2563eb;
            font-size: 21px;
        }

        /* INFO BOXES */

        .info-section {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-top: 25px;
        }

        .info-box {
            background: white;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            border: 1px solid #dbeafe;
            box-shadow: 0 3px 12px rgba(37,99,235,0.07);
        }

        .info-box .icon {
            font-size: 28px;
            margin-bottom: 8px;
        }

        .info-box h3 {
            margin: 5px 0;
            color: #1d4ed8;
            font-size: 16px;
        }

        .info-box p {
            margin: 5px 0 0;
            color: #64748b;
            font-size: 13px;
            line-height: 1.5;
        }

        /* NOTE */

        .note {
            margin-top: 25px;
            padding: 15px 18px;
            background: #fff7ed;
            border-left: 4px solid #f59e0b;
            border-radius: 6px;
            color: #78350f;
            font-size: 13px;
            line-height: 1.5;
        }

        /* FOOTER */

        footer {
            margin-top: 40px;
            padding: 25px;
            text-align: center;
            background: #1e3a8a;
            color: #dbeafe;
            font-size: 13px;
        }

        /* MOBILE */

        @media (max-width: 650px) {

            .header h1 {
                font-size: 25px;
            }

            .container {
                width: 94%;
                margin: 25px auto;
            }

            .card {
                padding: 22px;
            }

            .form-grid {
                grid-template-columns: 1fr;
            }

            .predict-btn {
                grid-column: auto;
            }

            .info-section {
                grid-template-columns: 1fr;
            }
        }

    </style>
</head>

<body>

    <!-- HEADER -->

    <div class="header">

        <h1>🩺 AI-Based Diabetes Prediction</h1>

        <p>
            Machine Learning Based Health Risk Analysis System
        </p>

    </div>


    <!-- MAIN CONTENT -->

    <div class="container">

        <div class="card">

            <h2>Patient Health Information</h2>

            <p class="card-description">
                Enter the patient's health information below
                to generate a diabetes risk prediction.
            </p>


            <form method="POST">

                <div class="form-grid">

                    <!-- AGE -->

                    <div class="form-group">

                        <label>Age</label>

                        <input
                            type="number"
                            name="age"
                            min="18"
                            max="100"
                            placeholder="Enter age"
                            required
                        >

                    </div>


                    <!-- BMI -->

                    <div class="form-group">

                        <label>BMI</label>

                        <input
                            type="number"
                            name="bmi"
                            min="10"
                            max="60"
                            step="0.1"
                            placeholder="e.g. 24.5"
                            required
                        >

                    </div>


                    <!-- PHYSICAL ACTIVITY -->

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


                    <!-- BLOOD PRESSURE -->

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


                    <!-- CHOLESTEROL -->

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


                    <!-- GLUCOSE -->

                    <div class="form-group">

                        <label>Glucose Level</label>

                        <input
                            type="number"
                            name="glucose"
                            min="50"
                            max="300"
                            placeholder="e.g. 100"
                            required
                        >

                    </div>


                    <!-- BUTTON -->

                    <button
                        type="submit"
                        class="predict-btn"
                    >
                        🔍 Predict Diabetes Risk
                    </button>

                </div>

            </form>


            <!-- RESULT -->

            {% if result %}

            <div class="result">

                <h2>{{ result }}</h2>

                <p>
                    Estimated Diabetes Probability:
                    <strong>{{ probability }}%</strong>
                </p>

            </div>

            {% endif %}

        </div>


        <!-- INFORMATION CARDS -->

        <div class="info-section">

            <div class="info-box">

                <div class="icon">🤖</div>

                <h3>AI Powered</h3>

                <p>
                    Uses a Random Forest machine-learning
                    model for prediction.
                </p>

            </div>


            <div class="info-box">

                <div class="icon">📊</div>

                <h3>6 Health Factors</h3>

                <p>
                    Analyzes age, BMI, activity, blood pressure,
                    cholesterol and glucose.
                </p>

            </div>


            <div class="info-box">

                <div class="icon">⚡</div>

                <h3>Fast Results</h3>

                <p>
                    Get a prediction immediately after
                    submitting the information.
                </p>

            </div>

        </div>


        <!-- DISCLAIMER -->

        <div class="note">

            <strong>Important:</strong>
            This is an educational machine-learning project
            and is not a medical diagnosis. Please consult a
            qualified healthcare professional for medical advice.

        </div>

    </div>


    <!-- FOOTER -->

    <footer>

        AI-Based Diabetes Prediction System
        <br><br>
        Built with Python • Flask • Scikit-learn

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

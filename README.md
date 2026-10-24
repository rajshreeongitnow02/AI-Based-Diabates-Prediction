# AI Based Diabates Prediction
Using supervised machine learning techniques, this project can be used in the medical field for healthcare professionals and individuals identify risk factors early, enabling timely preventive care and intervention.

Used FLASK and RENDER for live website

There is only 30% use of AI in this project and rest is human work. 
I have used AI to generate website's good colouring layout and for informative purposes like 
- how to publish website using 'RENDER' /
- what packages were needed to install for the execution in my PC

**File	Purpose**
train_model.py -	Loads data, preprocesses it, trains the Random Forest, evaluates it, and saves model artifacts.
app.py -	Runs the Flask web application and performs predictions using the saved artifacts.
diabetes_binary_health (1).csv -	Dataset used by the training script.
diabetes_random_forest.pkl -	Serialized trained Random Forest model.
diabetes_imputer.pkl -	Serialized median imputer used during preprocessing.
requirements.txt -	Lists Python dependencies required by the application/deployment.
README.md -	Project overview and usage documentation.


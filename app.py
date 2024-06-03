import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import warnings
from logger import get_logger

warnings.filterwarnings('ignore')

app = Flask(__name__)
logger = get_logger()

# Load the model from the file
model_path = 'models/final_model.pkl'
try:
    with open(model_path, 'rb') as file:
        pickled_model = pickle.load(file)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve form data with validation
        try:
            age = float(request.form['age'])
            sex = float(request.form['sex'])
            T3 = float(request.form['T3'])
            TT4 = float(request.form['TT4'])
            FTI = float(request.form['FTI'])
            onthyroxine = float(request.form['onthyroxine'])
            sick = float(request.form['sick'])
            queryhypothyroid = float(request.form['queryhypothyroid'])
            psych = float(request.form['psych'])
        except ValueError:
            logger.error("Invalid input: Non-numeric values provided.")
            return render_template('index.html', prediction_text='Error: Please enter valid numerical values.')

        # Log received form data
        logger.debug(f"Received form data: age={age}, sex={sex}, T3={T3}, TT4={TT4}, FTI={FTI}, "
                      f"onthyroxine={onthyroxine}, sick={sick}, queryhypothyroid={queryhypothyroid}, psych={psych}")

        # Create a DataFrame from form data
        values = {"age": age, "sex": sex, "T3": T3, "TT4": TT4, "FTI": FTI,
                  "onthyroxine": onthyroxine, "sick": sick,
                  "queryhypothyroid": queryhypothyroid, "psych": psych}
        df_transform = pd.DataFrame([values])

        # Ensure order of columns matches model training
        df_transform = df_transform[['age', 'sex', 'onthyroxine', 'sick',
                                     'queryhypothyroid', 'psych', 'T3', 'TT4', 'FTI']]

        # Convert DataFrame to numpy array for prediction
        arr = df_transform.to_numpy()

        # Log the array for prediction
        logger.debug(f"Array for prediction: {arr}")

        # Predict using the loaded model
        pred = pickled_model.predict(arr)

        # Determine result
        if pred == 0:
            res_val = 'compensated_hypothyroid'
        elif pred == 1:
            res_val = 'negative'
        elif pred == 2:
            res_val = 'primary_hypothyroid'
        else:
            res_val = 'secondary_hypothyroid'

        # logger.info(f"Prediction result: {res_val}")
        # return render_template('index.html', prediction_text=f'Result: {
        logger.info(f"Prediction result: {res_val}")
        return render_template('index.html', prediction_text=f'Result: {res_val}')
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
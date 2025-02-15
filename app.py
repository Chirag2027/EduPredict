from flask import Flask, request, render_template
import numpy as np
import pandas as pd 

from sklearn.preprocessing import StandardScaler   # I'll use pickle file
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)  # Entry point

app = application

# Route for Home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    # All work like getting the data nad prediction will be done here

    if request.method == 'GET':
        return render_template('home.html') 
        # In home.html, simple input data fields will be there

    # POST ->> capture the data, do data transformation
    else:
        # CustomData --> present in predict_pipeline
        data = CustomData(
            # Reading the values from the form
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))

        )

        # above data ko df me convert krna
        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        # results will be in the list format, hence returning results[0]
        return render_template('home.html', results = results[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)

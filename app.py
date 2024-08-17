from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
             Age=int(request.form.get('age')),
            TypeofContact=request.form.get('typeofcontact'),
            CityTier=int(request.form.get('citytier')),
            DurationOfPitch=float(request.form.get('durationofpitch')),
            Occupation=request.form.get('occupation'),
            Gender=request.form.get('gender'),
            NumberOfFollowups=int(request.form.get('numberoffollowups')),
            ProductPitched=request.form.get('productpitched'),
            PreferredPropertyStar=float(request.form.get('preferredpropertystar')),
            MaritalStatus=request.form.get('maritalstatus'),
            NumberOfTrips=int(request.form.get('numberoftrips')),
            Passport=int(request.form.get('passport')),
            PitchSatisfactionScore=int(request.form.get('pitchesatisfactionscore')),
            OwnCar=int(request.form.get('owncar')),
            Designation=request.form.get('designation'),
            MonthlyIncome=float(request.form.get('monthlyincome')),
            TotalVisiting=int(request.form.get('totalvisiting'))

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0")      
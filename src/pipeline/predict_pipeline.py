import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
                 Age: int,
                 TypeofContact: str,
                 CityTier: int,
                 DurationOfPitch: int,
                 Occupation: str,
                 Gender: str,
                 NumberOfFollowups: int,
                 ProductPitched: str,
                 PreferredPropertyStar: float,
                 MaritalStatus: str,
                 NumberOfTrips: int,
                 Passport: int,
                 PitchSatisfactionScore: int,
                 OwnCar: int,
                 Designation: str,
                 MonthlyIncome: float,
                 NumberOfPersonVisiting: int,
                 NumberOfChildrenVisiting:int):

        self.Age = Age
        self.TypeofContact = TypeofContact
        self.CityTier = CityTier
        self.DurationOfPitch = DurationOfPitch
        self.Occupation = Occupation
        self.Gender = Gender
        self.NumberOfFollowups = NumberOfFollowups
        self.ProductPitched = ProductPitched
        self.PreferredPropertyStar = PreferredPropertyStar
        self.MaritalStatus = MaritalStatus
        self.NumberOfTrips = NumberOfTrips
        self.Passport = Passport
        self.PitchSatisfactionScore = PitchSatisfactionScore
        self.OwnCar = OwnCar
        self.Designation = Designation
        self.MonthlyIncome = MonthlyIncome
        self.NumberOfPersonVisiting = NumberOfPersonVisiting
        self.NumberOfChildrenVisiting = NumberOfChildrenVisiting        

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Age": [self.Age],
                "TypeofContact": [self.TypeofContact],
                "CityTier": [self.CityTier],
                "DurationOfPitch": [self.DurationOfPitch],
                "Occupation": [self.Occupation],
                "Gender": [self.Gender],
                "NumberOfFollowups": [self.NumberOfFollowups],
                "ProductPitched": [self.ProductPitched],
                "PreferredPropertyStar": [self.PreferredPropertyStar],
                "MaritalStatus": [self.MaritalStatus],
                "NumberOfTrips": [self.NumberOfTrips],
                "Passport": [self.Passport],
                "PitchSatisfactionScore": [self.PitchSatisfactionScore],
                "OwnCar": [self.OwnCar],
                "Designation": [self.Designation],
                "MonthlyIncome": [self.MonthlyIncome],
                "NumberOfPersonVisiting": [self.NumberOfPersonVisiting],
                "NumberOfChildrenVisiting": [self.NumberOfChildrenVisiting],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

import pandas as pd
import numpy as np

def create_features(data):

    data["Application_Receipt_Date"] = pd.to_datetime(data["Application_Receipt_Date"], format= '%m/%d/%Y')
    data["Manager_DoB"] = pd.to_datetime(data["Manager_DoB"], format= '%m/%d/%Y')
    data["Applicant_BirthDate"] = pd.to_datetime(data["Applicant_BirthDate"], format= '%m/%d/%Y')

    print(data["Manager_DoB"].year)
    # data["Manager_Age"] = 
    # data = data.apply( lambda x: x["Manager_Age"] = x["Application_Receipt_Date"].to_datetime() , axis = 1)
    return data

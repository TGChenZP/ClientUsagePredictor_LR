import pandas as pd
import sys
import numpy as np
import pickle


WEEK_TO_INVESTIGATE = int(sys.argv[1])


# Reading in data from the correct weeks (backtracking). The backtracking process can occur because the choices of score/what week's predictor a particular prediction used is well documented in the multiple csv files
predictions = pd.read_csv(f'./History/Week {WEEK_TO_INVESTIGATE-1}/Predictions.csv')

data = pd.read_csv(f'./History/Week {WEEK_TO_INVESTIGATE+1}/Partial_Output/_3_combined_cleaned_data.csv')

with open(f'./Models/attbs.pickle', 'rb') as f:
    attrbs = pickle.load(f)

# Prepare the observed data and give them labels of "Increase", "Decrease" and "Normal" and output into file
drop = '-26'
nweeks = '11'
r = 'r2'

for col in attrbs:
    test = pd.read_csv(f'./History/Week {WEEK_TO_INVESTIGATE+1}/PreparedData/S_D_{drop}_{col}_{nweeks}.csv')
    test = test[test['Week'] == WEEK_TO_INVESTIGATE]
    
    boolean = test.Domain.isin(list(predictions['Domain']))
    test = test[boolean]
    test.index = test['Domain']
    
    tmp = [test.loc[domain]['Target'] if domain in list(test.index) else 0 for domain in list(predictions['Domain'])]
    
    predictions.insert(len(predictions.columns), f'{col}O', tmp, True)
    
predictions['ObsScore'] = [0 for i in range(len(predictions))]

if r == 'r':
    for col in attrbs:
        
        with open(f'./Models/{col}.pickle', 'rb') as f:
            obj = pickle.load(f)    
    
        R = obj[1]

        predictions['ObsScore'] = predictions['ObsScore'] + R*predictions[f'{col}O']


elif r == 'r2':
    for col in attrbs:
        
        with open(f'./Models/{col}.pickle', 'rb') as f:
            obj = pickle.load(f)    

        R = obj[1]

        predictions['ObsScore'] = predictions['ObsScore'] + (R**2)*predictions[f'{col}O']
        
else:
    for col in attrbs:
        predictions['ObsScore'] = predictions['ObsScore'] + predictions[f'{col}O']

final = pd.DataFrame()

predictions['ObsScore'] = predictions['ObsScore'].replace((np.nan), 0)
O95 = np.quantile(predictions['ObsScore'], .95)
O05 = np.quantile(predictions['ObsScore'], .05)

predictions['Observations'] = ['Increase' if predictions.loc[i]['ObsScore'] > O95 else 'Decrease' if predictions.loc[i]['ObsScore'] < O05 else 'Normal' for i in range(len(predictions))]

predictions = predictions[['ID', 'Client id', 'Client name', 'Domain', 'Predictions', 'Observations']]

predictions.to_csv(f'./History/Week {WEEK_TO_INVESTIGATE-1}/Retrospective.csv', index = False)





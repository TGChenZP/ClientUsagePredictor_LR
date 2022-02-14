### CLIENT USAGE PREDICTOR - prediction script
### Code produced by Lang (Ron) Chen July-December 2021 for Lucidity Software
""" Wrangles raw data of recent weeks, appends it to the previous wrangled data and outputs predictions for upcoming week """

# Input:
#    Argument 1: the start date for the new data (only include data from after this date); logically should be a Monday. If this script is run as preferably (weekly), then this argument should be the Monday of the previous week. Must be in form ‘[d]d/[m]m/yyyy’ (If day or month single digit can input just as single digit). 
#         please ensure date is valid and is after the FIRSTDATE (by default July 3rd 2017)
#    Argument 2: a cut-off date for data; logically should be a Sunday Must be in form ‘[d]d/[m]m/yyyy’ (If day or month single digit can input just as single digit). 
#         please ensure date is valid and is after the FIRSTDATE (by default July 3rd 2017) and after the date of Argument 1

#     Initial raw data files need to be stored in directory './History/Week {this week week number}/Data’. 
#     -File names must be ‘action.csv’, ‘assets.csv’, ‘competency_record.csv’, ‘form_record.csv’, ‘form_template.csv’, ‘incident.csv’, ‘users.csv’, associated with the data of the corresponding filename
#     -Each csv must include columns that include ‘domain’ for company name, and ‘created_at’. 
#     -There should be no other tester domains in the data apart from ‘demo’, ‘demo_2’ and ‘cruse’

#     -the dates for all files should be in form of yyyy-mm-dd.
#     *if the form of these are different then need to edit PART 2 in the script. 


# Output: several partial outputs - partial outputs of wrangled data and prediction output exported to various directories including the home directory (relatively '.') and the History directory (into the relevent week)





# PART 0: IMPORTING LIBRARIES

import sys
import os

import pandas as pd
import numpy as np

import math as m

from sklearn import linear_model
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import KFold

import pickle





# PART 1: CREATE DATE FILE
# Slightly different from PART 1 of InitialTraining.py's part 1 because it also needs to account for a subset of dates (from just the date of argument 1 to the date of argument 2)

# Reads in 'cut-off date' for data to be used to train 
datastartdate = sys.argv[1]
currdate = sys.argv[2]


# Assumes that date format in 'dd/mm/yyyy' format
cutoffday = int(currdate.split('/')[0])
cutoffmonth = int(currdate.split('/')[1])
cutoffyear = int(currdate.split('/')[2])

def datejoin(day, month, year):
    """For joining up three numbers into a date"""
    return (f'{str(day)}/{str(month)}/{str(year)}')


def leapyear(year):
    """For determining whether a year is a leap year"""
    if year % 4 == 0:
        if year% 100 == 0:
            if year%400 == 0:
                return True
            else:
                return False
        else:
            return True
        
    else:
        return False


# Creates a dictionary matching each day to a week number (counting Week of July 3rd 2021 as Week 1)    
#### FUTURE CHANGE: if wish to include data earlier than Monday July 3rd 2017, change the magic string FIRSTDATE
FIRSTDATE = '03/07/2017'
firstdateday = int(FIRSTDATE.split('/')[0])
firstdatemonth = int(FIRSTDATE.split('/')[1])
firstdateyear = int(FIRSTDATE.split('/')[2])

datastartdateday = int(datastartdate.split('/')[0])
datastartdatemonth = int(datastartdate.split('/')[1])
datastartdateyear = int(datastartdate.split('/')[2])

days = [29, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]    
months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
years = range(2017, cutoffyear+1)


datematchweek = dict()
minidatematchweek = dict()
week = 1
count = 0
for year in years:
    for month in months:
        
        if (year == firstdateyear and month < firstdatemonth) or (year == cutoffyear and month > cutoffmonth):
            continue
        
        if month == 2 and leapyear(year):
            indexmonth = 0
            
        else:
            indexmonth = month
        
        for day in range(1, days[indexmonth]+1):
            if (year == firstdateyear and month == firstdatemonth and day < firstdateday) or (year == cutoffyear and month == cutoffmonth and day > cutoffday):
                continue
            
            count += 1
            
            if count == 8:
                count = 1
                week += 1
            
            date = datejoin(day, month, year)
            
            datematchweek[date] = week
            
            if year > datastartdateyear:
                
                minidatematchweek[date] = week
                
            elif year == datastartdateyear:
                
                if month > datastartdatemonth:
                    
                    minidatematchweek[date] = week
                    
                elif month == datastartdatemonth and day >= datastartdateday:  
                            
                    minidatematchweek[date] = week

dates = list(datematchweek.keys())
weekno = list(datematchweek.values())

# Make the dictionary of dates to week number into a dataframe
DatesToWeek_DF = pd.DataFrame({'dates': dates, 'weekno':weekno})

DatesToWeek_DF.to_csv('DateMatchWeek.csv')

# Record the week number of the cut-off date
thisweek = max(weekno)

# create a subset just for wrangling this week's data 
minidates = list(minidatematchweek.keys())
miniweekno = list(minidatematchweek.values())

MiniDatesToWeek_DF = pd.DataFrame({'dates': minidates, 'weekno':miniweekno})

datastartweek = min(miniweekno)







# PART 2: WRANGLING ORIGINAL DATA

#### FUTURE UPDATE: towranglelist1 stores records which have dates in format [d]d/[m]m/yyyy; towranglelist2 stores records which have dates in format yyyy-mm-dd. Need to update lists accordingly
towranglelist1 = [] #### FUTURE UPDATE: because all data at the time of writing came from Quicksight, all their formats were yyyy-mm-dd so all were put into towranglelist2
towranglelist2 = ['action.csv', 'competency_record.csv', 'form_record.csv', 'incident.csv', 'users.csv', 'assets.csv', 'form_template.csv']


def wrangle(filename, datedata, mode):
    """ cleans the file. 4 modes for four different ways to clean the data - all pretty similar except mode 3 and 4 selects users of particular hr types, and mode 2 deals with dates of a different format """
    data = pd.read_csv(f"./History/Week {thisweek}/Data/{filename}")
    domain = list(data['domain'])

    # First drop: get rid of rows from domains demo and demo_2
    if mode in [1,2]:
        droplist = []
        for i in range(len(domain)):
            if domain[i] in ['demo', 'demo_2', 'cruse']:
                droplist.append(i)


    data = data.drop(droplist)

    # re-setup date dictionary from the DataFrame
    dates = list(datedata['dates'])
    weekno = list(datedata['weekno'])
    datematchdict = dict()
    for i in range(len(dates)):
        datematchdict[dates[i]] = weekno[i]

    # Second drop: clean out rows whose dates are not within startdate and cutoffdate
    #### FUTURE CHANGE: this step takes quite a lot of time - could be area to improve algorithmically
    data.index = (range(0, len(list(data['created_at'])))) #re-do index after dropping demo and demo_2
    actdate = list(data['created_at'])

    # If any data happens to have date in format "dd-mm-yyyy" then need to put file in towranglelist2. Else put in towranglelist1. Note dates should be in format "[d]d/[m]m/yyyy"
    newdroplist = []
    if mode in [2]:
        def transform_date(inputdate):
            """ helper function to transform date in format of dd-mm-yyyy into [d]d/[m]m/yyyy which is what datedata produced in PART 1 stores  """
            splitted = inputdate.split('-')
            if int(splitted[1]) < 10:
                month = splitted[1][1]
            else:
                month = splitted[1]

            if int(splitted[2]) < 10:
                day = splitted[2][1]
            else:
                day = splitted[2]

            return f'{day}/{month}/{splitted[0]}'

        for i in range(len(actdate)):
            if transform_date(actdate[i].split()[0]) not in dates:
                newdroplist.append(i)

    else:
        for i in range(len(actdate)):
            if actdate[i].split()[0] not in dates:
                newdroplist.append(i)

    data = data.drop(newdroplist) # drop the rows of data whose dates are not between startdate and cutoffdate

    actdate = list(data['created_at']) #reread the date created column now that we've dropped some rows

    newdomain = list(data['domain'])
    # get a new list matching each action to the week that they were done in
    actweekno = list()

    if mode in [2]:
        for i in range(len(actdate)):
            actweekno.append(datematchdict[transform_date(actdate[i].split()[0])])
    else:
        for i in range(len(actdate)):
            actweekno.append(datematchdict[actdate[i].split()[0]]) # use [0] because string also contains hour:minute:second

    # At this point, now have two lists newdomain and actweekno: in the former the ith value is the domain of the ith row, and the latter the ith value is the relative week since FIRSTSTARTDATE that the ith row was created in. Now just count them up 

    # count up the numbers of actions this week by domain and week
    groupup = dict()
    for i in range(len(actweekno)):
        if f'{newdomain[i]} {actweekno[i]}' in groupup:
            groupup[f'{newdomain[i]} {actweekno[i]}'] += 1
        else:
            groupup[f'{newdomain[i]} {actweekno[i]}'] = 1

    groupupkey = list(groupup.keys())
    groupupval = list(groupup.values())

    # create lists that contain just domain name and week number
    out1 = list()
    out2 = list()

    for i in range(len(groupupkey)):
        out1.append(groupupkey[i].split()[0])
        out2.append(groupupkey[i].split()[1])

    # export the wrangled file as a csv (each of these files are wrangled version of the raw data files (of each of the client's recorded activity in lucidity) in terms of counts per week per domain)
    out = pd.DataFrame({'Domain': out1, 'Week': out2, 'COUNT': groupupval})

    if mode in [1,2]:
        out.to_csv(f'./History/Week {thisweek}/Partial_Output/_2_{filename.split(".")[0]}_clean.csv', index = False)

# OS housekeeping and running each of the files through wrangle()
if not os.path.exists(f'./History/Week {thisweek}/Partial_Output'):
    os.mkdir(f'./History/Week {thisweek}/Partial_Output')

for file in towranglelist1:
    wrangle(file, MiniDatesToWeek_DF, 1)

for file in towranglelist2:
    wrangle(file, MiniDatesToWeek_DF, 2)





# PART 3: COMBINE PREVIOUSLY WRANGLED DATAFRAMES INTO ONE (FILLING IN WEEKS WITH NO ACTIVITY)

# import all cleaned data
asset = pd.read_csv(f'./History/Week {thisweek}/Partial_Output/_2_assets_clean.csv')
actions = pd.read_csv(f'./History/Week {thisweek}/Partial_Output/_2_action_clean.csv')
competency = pd.read_csv(f'./History/Week {thisweek}/Partial_Output/_2_competency_record_clean.csv')
form_record = pd.read_csv(f'./History/Week {thisweek}/Partial_Output/_2_form_record_clean.csv')
form_templates = pd.read_csv(f'./History/Week {thisweek}/Partial_Output/_2_form_template_clean.csv')
incidents = pd.read_csv(f'./History/Week {thisweek}/Partial_Output/_2_incident_clean.csv')
users = pd.read_csv(f'./History/Week {thisweek}/Partial_Output/_2_users_clean.csv')

# Find a set of the domain names - for finding the "earliest recorded date" of activity/usage for each
set1 = set(asset['Domain'])
set2 = set(actions['Domain'])
set3 = set(competency['Domain'])
set4 = set(form_record['Domain'])
set5 = set(form_templates['Domain'])
set6 = set(incidents['Domain'])
set7 = set(users['Domain'])

fullset = set1.union(set2).union(set3).union(set4).union(set5).union(set6).union(set7)

iteration = [asset, actions, competency, form_record, form_templates, incidents, users]
newiteration = [asset, actions, competency, form_record, form_templates, incidents, users]

# Find first week recorded and put them in a dictionary of (key: value) = (domain name: first week of activity)
startweek = dict()

# read in data file _3_ from last week (for concatenating this week's data onto)
olddata = pd.read_csv(f'./History/Week {datastartweek-1}/Partial_Output/_3_combined_cleaned_data.csv')
olddoms = list(set(olddata['Domain']).union(fullset))

# Create a template for recording the data (now fill out gaps between start week and week 216 where there is 0 data)
combineddatatemplate = dict()

# first initiate a blank dictionary with all weeks from the week of argument 1 of this script (the date from which we should start counting in the data)
# this is slightly different from InitialTraining because we are adding new weeks on top of last time's data, so even if no activity at all these weeks we still need to include it.
for dom in olddoms:
    for j in range(datastartweek, thisweek+1):
        combineddatatemplate[f'{dom} {j}'] = 0

# create blank copies of this initialised template dictionary, and fill them in based on counts from the output of PART 2
assetcomb = combineddatatemplate.copy()
actionscomb = combineddatatemplate.copy()
competencycomb = combineddatatemplate.copy()
form_recordcomb = combineddatatemplate.copy()
form_templatescomb = combineddatatemplate.copy()
incidentscomb = combineddatatemplate.copy()
userscomb = combineddatatemplate.copy()


# dictlist = [assetcomb, actionscomb, competencycomb, form_recordcomb, form_templatescomb, incidentscomb, userscomb, users_inductcomb, users_norm_empcomb]
dictlist = [assetcomb, actionscomb, competencycomb, form_recordcomb, form_templatescomb, incidentscomb, userscomb]

# Now fill in the details where there are records (because all dictionary slots initialised, only need to repalce data for weeks where there was a count recorded, and all other are fine to be left untouched - just ends up being 0)
for k in range(len(dictlist)):
    dom = list(newiteration[k]['Domain'])
    week = list(newiteration[k]['Week'])
    count = list(newiteration[k]['COUNT'])

    for i in range(len(dom)):
        dictlist[k][f'{dom[i]} {week[i]}'] = count[i]

uniqueid = list(assetcomb.keys())
assetcount = list(assetcomb.values())
actioncount = list(actionscomb.values())
competencycount = list(competencycomb.values())
form_recordcount = list(form_recordcomb.values())
form_templatescount = list(form_templatescomb.values())
incidentscount = list(incidentscomb.values())
userscount = list(userscomb.values())

# create two more lists that contain just domain and just week - maximises chances of making future wrangling easier
doms = []
weekss = []

for i in range(len(uniqueid)):
    doms.append(uniqueid[i].split()[0])
    weekss.append(uniqueid[i].split()[1])

# Create a new column for counting the number of weeks since particular company started at Lucidity
selfweeks = []
count = 0

# make all the selfweek number for this new data an impossible value of -1
prev = doms[0]
for i in range(len(doms)):
    selfweeks.append(-1)

# turn it into one dataframe and concatanate onto old data
out = pd.DataFrame({'ID': uniqueid, 'Domain': doms, 'Week': weekss, 'Selfweeks': selfweeks,
                    'Assets': assetcount, 'Actions': actioncount, 'Competency': competencycount, 
                    'Form_record': form_recordcount, 'Form_template': form_templatescount,
                   'Incident': incidentscount, 'Users': userscount})

combineddata = pd.concat([olddata, out])
combineddata = combineddata.sort_values(['Domain', 'Week'], axis=0) # Sorting the dataframe by domain and weeks

selfweeks = list(combineddata['Selfweeks'])
domain = list(combineddata['Domain'])

# for any self weeks that are -1 (newly added on), just add one onto it from previous week. (it works since the data is sorted)
prev = domain[0]
for i in range(len(selfweeks)):

    if domain[i] == prev and selfweeks[i] == -1:
        selfweeks[i] = selfweeks[i-1] + 1

    elif domain[i] != prev and selfweeks[i] == -1:
        selfweeks[i] = 1

    prev = domain[i]

# replacing the column 'Selfweeks' with correct data        
combineddata['Selfweeks'] = selfweeks

# This file now has counts of all activities grouped by week by client/domain, sorted by domian and clients. 
combineddata.to_csv(f'./History/Week {thisweek}/Partial_Output/_3_combined_cleaned_data.csv', index = False)



    
    
# PART 4: MANIPULATE DATA
def manipdata(data, NWEEKS, attrb, meth, maniptype, discard, thisweek):
    
    metadata = ['ID', 'Domain', 'Week', 'Selfweeks']
    masterlist = [list() for i in range(NWEEKS+5)]
    
    skip = 0
    if discard:
        skip = 26
    
    
    for i in range(len(data)):
        if data.loc[i]['Selfweeks'] > NWEEKS+skip:
            
            for j in range(len(metadata)):
                masterlist[j].append(data.loc[i][metadata[j]])
                
            for j in range(4, NWEEKS+4):
                masterlist[j].append(data.loc[i-(j-4)][attrb])
            
            masterlist[NWEEKS+4].append(data.loc[i][f'{attrb}T'])
    
    out = pd.DataFrame()
        
    for i in range(NWEEKS+5):
        if i < 4:
            out.insert(i, metadata[i], masterlist[i])
        elif i == NWEEKS + 4:
            out.insert(i, 'Target', masterlist[i])
        else:
            out.insert(i, f'{i-4}', masterlist[i])
    
    if not os.path.exists(f'./History/Week {thisweek}/PreparedData'):
        os.mkdir(f'./History/Week {thisweek}/PreparedData')
    
    out.to_csv(f'./History/Week {thisweek}/PreparedData/{maniptype}_{meth}_{attrb}_{NWEEKS}.csv', index = False)


discard = False
meth = '-26'

if '-26' in meth:
    combineddata = combineddata[combineddata['Selfweeks'] > 26]
    combineddata.index = range(len(combineddata))

    discard = True

data = pd.DataFrame()
for domain, tmp in combineddata.groupby('Domain'):
    tmp.index = range(len(tmp))
        
    for i in range(len(tmp)-1, -1, -1):
        if ((tmp.loc[i]['Assets']) | (tmp.loc[i]['Actions']) | (tmp.loc[i]['Competency']) |
                (tmp.loc[i]['Form_record']) | (tmp.loc[i]['Form_template']) |
                (tmp.loc[i]['Incident']) | (tmp.loc[i]['Users'])):
            break
        
    data = pd.concat([data, tmp[0:i+1]])

stdData = pd.DataFrame()
    
for domain, compData in data.groupby('Domain'):
    
    with open(f'./Standardisers/{domain}.pickle', 'rb') as f:
              scaler = pickle.load(f)
        
    compData[['Assets', 'Actions', 'Competency',
        'Form_record', 'Form_template', 'Incident', 'Users']] = scaler.transform(compData[['Assets', 'Actions', 'Competency',
        'Form_record', 'Form_template', 'Incident', 'Users']])

    stdData = pd.concat([stdData, compData])


with open(f'./Models/attbs.pickle', 'rb') as f:
    attrbs = pickle.load(f)

    
for colName in stdData.columns[4:]:

        target = list()

        for domain, compData in stdData.groupby('Domain'):
            
            compData.index = compData['Selfweeks']
            
            index = compData.index
            
            out = [(compData.loc[i+1][colName]-compData.loc[i][colName]) if (i+1 in index)
                   else np.nan for i in index]
            target.extend(out)
        
        stdData[f'{colName}T'] = target

stdData['Week'].astype(int)
stdData = stdData.sort_values(['Domain', 'Week'])
stdData.index = range(len(stdData))
        
for attrb in attrbs:
    for nWeeks in [11]:
        manipdata(stdData, nWeeks, attrb, meth, 'S_D', True, thisweek)



        
        
# PART 5: TRAIN LR MODELS AND TESTING        
cols = ['Assets', 'Actions', 'Competency',
       'Form_record', 'Form_template', 'Incident', 'Users']

r = 'r2'
nweeks = 11
drop = '-26'


# Training
models = dict()

# Test data
test = pd.read_csv(f'./History/Week {thisweek}/PreparedData/S_D_{drop}_Assets_{nweeks}.csv')
test = test[test['Week'] == thisweek]
out = test[test.columns[0:4]]

for col in attrbs:
    test = pd.read_csv(f'./History/Week {thisweek}/PreparedData/S_D_{drop}_{col}_{nweeks}.csv')
    test = test[test['Week'] == thisweek]

    xTest = test[test.columns[4:-1]]

    with open(f'./Models/{col}.pickle', 'rb') as f:
        obj = pickle.load(f)    

    lm = obj[0]

    yPred = lm.predict(xTest)

    out[f'{col}P'] = yPred

out['PredScore'] = [0 for i in range(len(out))]

if r == 'r2':
    for col in attrbs:
        with open(f'./Models/{col}.pickle', 'rb') as f:
            obj = pickle.load(f)    

        R = obj[1]

        out['PredScore'] = out['PredScore'] + (R**2)*out[f'{col}P']

else:
    for col in attrbs:
        out['PredScore'] = out['PredScore'] + out[f'{col}P']

predictions = pd.DataFrame()
for week, data in out.groupby(['Week']):
    data.index = range(len(data))

    P95 = np.quantile(data['PredScore'], .95)
    P05 = np.quantile(data['PredScore'], .05)

    data['Predictions'] = ['Increase' if data.loc[i]['PredScore'] > P95 else 'Decrease' if data.loc[i]['PredScore'] < P05 else 'Normal' for i in range(len(data))]

    predictions = pd.concat([predictions, data])

            



# PART 6: Make Predictions
predictions = predictions.sort_values(['Domain'], axis=0)

# adding on the Client and Client code for easier mapping later on
match = pd.read_excel('./Mapping.xlsx')

finaldom = list(predictions['Domain'])

dommatchdict1 = dict() # dictionary for storing client name
dommatchdict2 = dict() # dictionary for storing client id

for i in range(len(finaldom)):
    dommatchdict1[finaldom[i]] = ''
    dommatchdict2[finaldom[i]] = ''

domain1 = list(match['Domain1'])
clientname = list(match['Client Code'])
clientid = list(match['Client'])
for i in range(len(domain1)):
    if domain1[i] in dommatchdict1:
        dommatchdict1[domain1[i]] = clientname[i]
        dommatchdict2[domain1[i]] = clientid[i]

dommatchlist1 = list(dommatchdict1.values())
dommatchlist2 = list(dommatchdict2.values())

predictions.insert(1, "Client id", dommatchlist1, True)
predictions.insert(2, "Client name", dommatchlist2, True)

predictions = predictions.sort_values('PredScore')

predictions.to_csv(f'./History/Week {thisweek}/PredictionDetails.csv', index = False)

out = predictions[predictions.columns[0:4]]
out['Predictions'] = predictions['Predictions']

out.to_csv('./Predictions.csv', index = False)
out.to_csv(f'./History/Week {thisweek}/Predictions.csv', index = False)

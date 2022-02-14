### CLIENT USAGE PREDICTOR (LR) - initial training script
### Code produced by Lang (Ron) Chen Jul-Dec 2021 for Lucidity Software
""" Wrangles initial raw data and outputs predictor objects for predicting the client usage trend for the upcoming week """

# Input: 
#    Argument 1: a cut-off date for data; logically should be a Sunday. Must be in form ‘[d]d/[m]m/yyyy’ (If day or month single digit can input just as single digit). 
#         please ensure date is valid and is after the FIRSTDATE (by default July 3rd 2017)

#     Initial raw data files need to be stored in directory ‘./Data’. 
#     -File names must be ‘action.csv’, ‘assets.csv’, ‘competency_record.csv’, ‘form_record.csv’, ‘form_template.csv’, ‘incident.csv’, ‘users.csv’, associated with the data of the corresponding filename
#     -Each csv must include columns that include ‘domain’ for company name, and ‘created_at’. 
#     -There should be no other tester domains in the data apart from ‘demo’, ‘demo_2’ and ‘cruse’

#     -the dates for 'action.csv', 'competency_record.csv', 'form_record.csv', 'incident.csv', 'users.csv' should be in form of [d]d/[m]m/yyyy
#     -the dates for 'assets.csv', 'form_template.csv' should be in form of yyyy-mm-dd.
#     *if the form of these are different then need to edit PART 2 in the script. 


# Output: several partial outputs - partial outputs of wrangled data, Linear Regression objects (.pickle), Statistics.csv, exported to various directories including the home directory (relatively '..'), the History directory (into the relevent week) and the current directory 





# PART 0: IMPORTING LIBRARIES

import sys
import os

import pandas as pd
import numpy as np

import math as m

from sklearn import linear_model
from sklearn import preprocessing
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import pickle





# PART 1: CREATE DATE FILE

# Reads in 'cut-off date' for data to be used to train 
currdate = sys.argv[1]

# Assumes that date format in 'dd/mm/yyyy' format
cutoffday = int(currdate.split('/')[0])
cutoffmonth = int(currdate.split('/')[1])
cutoffyear = int(currdate.split('/')[2])

def datejoin(day, month, year):
    """ For joining up three numbers into a date"""
    return (f"{str(day)}/{str(month)}/{str(year)}")


def leapyear(year):
    """ For determining whether a year is a leap year"""
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

days = [29, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]    
months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
years = range(2017, cutoffyear+1)


datematchweek = dict()
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

dates = list(datematchweek.keys())
weekno = list(datematchweek.values())

# Make the dictionary of dates to week number into a dataframe
DatesToWeek_DF = pd.DataFrame({'dates': dates, 'weekno':weekno})

DatesToWeek_DF.to_csv('../DateMatchWeek.csv')

# Record the week number of the cut-off date
thisweek = max(weekno)





# # PART 2: WRANGLING ORIGINAL DATA

#### FUTURE UPDATE: towranglelist1 stores records which have dates in format [d]d/[m]m/yyyy; towranglelist2 stores records which have dates in format yyyy-mm-dd. Need to update lists accordingly
towranglelist1 = ['action.csv', 'competency_record.csv', 'form_record.csv', 'incident.csv', 'users.csv']
towranglelist2 = ['assets.csv', 'form_template.csv']

def wrangle(filename, datedata, mode):
    """ cleans the file. 4 modes for four different ways to clean the data - all pretty similar except mode 3 and 4 selects users of particular hr types, and mode 2 deals with dates of a different format """
    data = pd.read_csv(f"./Data/{filename}")
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
    if mode == 2:
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
    
    if mode == 2:
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
        out.to_csv(f'./Partial_Output/_2_{filename.split(".")[0]}_clean.csv', index = False)
        out.to_csv(f'../History/Week {thisweek}/Partial_Output/_2_{filename.split(".")[0]}_clean.csv', index = False)

# OS housekeeping and running each of the files through wrangle()
if not os.path.exists('./Partial_Output'):
    os.mkdir('./Partial_Output')

if not os.path.exists(f'../History/Week {thisweek}/Partial_Output'):
    os.makedirs(f'../History/Week {thisweek}/Partial_Output')
        
for file in towranglelist1:
    wrangle(file, DatesToWeek_DF, 1)
    
for file in towranglelist2:
    wrangle(file, DatesToWeek_DF, 2)


    

    
# PART 3: COMBINE PREVIOUSLY WRANGLED DATAFRAMES INTO ONE (FILLING IN WEEKS WITH NO ACTIVITY)

# import all cleaned data
asset = pd.read_csv('./Partial_Output/_2_assets_clean.csv')
actions = pd.read_csv('./Partial_Output/_2_action_clean.csv')
competency = pd.read_csv('./Partial_Output/_2_competency_record_clean.csv')
form_record = pd.read_csv('./Partial_Output/_2_form_record_clean.csv')
form_templates = pd.read_csv('./Partial_Output/_2_form_template_clean.csv')
incidents = pd.read_csv('./Partial_Output/_2_incident_clean.csv')
users = pd.read_csv('./Partial_Output/_2_users_clean.csv')

# Find a set of the domain names - for finding the "earliest recorded date" of activity/usage for each
set1 = set(asset['Domain'])
set2 = set(actions['Domain'])
set3 = set(competency['Domain'])
set4 = set(form_record['Domain'])
set5 = set(form_templates['Domain'])
set6 = set(incidents['Domain'])
set7 = set(users['Domain'])

fullset = set1.union(set2).union(set3).union(set4).union(set5).union(set6).union(set7)
fullsetlist = list(fullset) # Now have a full set of the domains
fullsetlist.sort()

iteration = [asset, actions, competency, form_record, form_templates, incidents, users]
newiteration = [asset, actions, competency, form_record, form_templates, incidents, users]

# Find first week recorded and put them in a dictionary of (key: value) = (domain name: first week of activity)
startweek = dict()

for data in newiteration:
    dom = list(data['Domain'])
    week = list(data['Week'])
    count = list(data['COUNT'])
    
    for i in range(len(dom)):
        if dom[i] in startweek:
            if week[i] < startweek[dom[i]]:
                startweek[dom[i]] = week[i]
        else:
            startweek[dom[i]] = week[i]
            
startweeklist = list(startweek.items())
startweeklist.sort()

# Create a template for recording the data (now fill out gaps between start week and week 216 where there is 0 data)
combineddatatemplate = dict()

# first initiate a blank dictionary with all weeks from first week of activity to cutoffdate's week
for i in range(len(startweeklist)):
    for j in range(startweeklist[i][1], thisweek+1):
        combineddatatemplate[f'{startweeklist[i][0]} {j}'] = 0

# create blank copies of this initialised template dictionary, and fill them in based on counts from the output of PART 2
assetcomb = combineddatatemplate.copy()
actionscomb = combineddatatemplate.copy()
competencycomb = combineddatatemplate.copy()
form_recordcomb = combineddatatemplate.copy()
form_templatescomb = combineddatatemplate.copy()
incidentscomb = combineddatatemplate.copy()
userscomb = combineddatatemplate.copy()

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

# (logic of loop fairly simple - if domain column runs into new company then reset the count)
prev = doms[0]
for i in range(len(doms)):
    if doms[i] == prev:
        count += 1
        selfweeks.append(count)
    else:
        count = 1
        selfweeks.append(count)
    prev = doms[i]


# turn it into one dataframe and output

out = pd.DataFrame({'ID': uniqueid, 'Domain': doms, 'Week': weekss, 'Selfweeks': selfweeks,
                    'Assets': assetcount, 'Actions': actioncount, 'Competency': competencycount, 
                    'Form_record': form_recordcount, 'Form_template': form_templatescount,
                   'Incident': incidentscount, 'Users': userscount})

# This file now has counts of all activities grouped by week by client/domain, sorted by domian and clients.  
out.to_csv("./Partial_Output/_3_combined_cleaned_data.csv", index = False)
out.to_csv(f"../History/Week {thisweek}/Partial_Output/_3_combined_cleaned_data.csv", index = False)





# PART 4: MANIPULATE DATA

if not os.path.exists('./PreparedData'):
    os.mkdir('./PreparedData')

if not os.path.exists('./SplitData'):
    os.mkdir('./SplitData')

    
if not os.path.exists('../Standardisers'):
    os.mkdir('../Standardisers')

if not os.path.exists('../Models'):
    os.mkdir('../Models')

if not os.path.exists('../History/Week 215/PreparedData'):
    os.mkdir('../History/Week 215/PreparedData')
    

    
def manipdata(data, NWEEKS, attrb, meth, maniptype, discard):
    """ manipulates data into desired time series form """
    
    metadata = ['ID', 'Domain', 'Week', 'Selfweeks']
    masterlist = [list() for i in range(NWEEKS+5)]
    
    skip = 0
    if discard:
        skip = 26
    
    
    for i in range(len(data)):
        if data.loc[i]['Selfweeks'] > NWEEKS+skip and i+1 < len(data) and data.loc[i]['Domain'] == data.loc[i+1]['Domain']:
            
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
    
    out.to_csv(f'./PreparedData/{maniptype}_{meth}_{attrb}_{NWEEKS}.csv', index = False)
    out.to_csv(f'../History/Week 215/PreparedData/{maniptype}_{meth}_{attrb}_{NWEEKS}.csv', index = False)



#Drop first 26 self weeks    
meth = '-26'

df = pd.read_csv('./Partial_Output/_3_combined_cleaned_data.csv')

discard = False

if '-26' in meth:
    df = df[df['Selfweeks'] > 26]
    df.index = range(len(df))

    discard = True

data = pd.DataFrame()
for domain, tmp in df.groupby('Domain'):
    tmp.index = range(len(tmp))

    for i in range(len(tmp)-1, -1, -1):
        if ((tmp.loc[i]['Assets']) | (tmp.loc[i]['Actions']) | (tmp.loc[i]['Competency']) |
                (tmp.loc[i]['Form_record']) | (tmp.loc[i]['Form_template']) |
                (tmp.loc[i]['Incident']) | (tmp.loc[i]['Users'])):
            break

    data = pd.concat([data, tmp[0:i+1]])

# Drop end zeroes   
stdData = pd.DataFrame()

for domain, compData in data.groupby('Domain'):

    compDataX = compData[['Assets', 'Actions', 'Competency',
       'Form_record', 'Form_template', 'Incident', 'Users']]

    scaler = preprocessing.StandardScaler().fit(compDataX)

    with open(f'../Standardisers/{domain}.pickle', 'wb') as f:
        pickle.dump(scaler, f)

    compData[['Assets', 'Actions', 'Competency',
       'Form_record', 'Form_template', 'Incident', 'Users']] = scaler.transform(compDataX)

    stdData = pd.concat([stdData, compData])

# Create Targets
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

for attrb in stdData.columns[4:11]:
    for nWeeks in [11]:
        manipdata(stdData, nWeeks, attrb, meth, 'S_D', discard)
    

    
    
    
# PART 5: TRAIN TEST SPLIT

filelist = os.listdir("./PreparedData")
filelist.sort()
for file in filelist:
    if file[0] != '.':
        stdData = pd.read_csv(f"./PreparedData/{file}")

        trainWeeks, testWeeks = train_test_split(range(12, 215), train_size = 0.8, test_size = 0.2, random_state = 42)
        testBool = stdData.Week.isin(testWeeks)
        testData = stdData[testBool]

        trainBool = stdData.Week.isin(trainWeeks)
        trainData = stdData[trainBool]

        testData.to_csv(f'./SplitData/{file.strip(".csv")}_Test.csv', index = False)
        trainData.to_csv(f'./SplitData/{file.strip(".csv")}_Train.csv', index = False)

    


# PART 6: TRAIN LR MODELS AND TESTING
cols = ['Assets', 'Actions', 'Competency',
       'Form_record', 'Form_template', 'Incident', 'Users']

fsval = 0.4
r = 'r2'
drop = '-26'
nweeks = 11



# Training
models = dict()

for attb in cols:   
    df = pd.read_csv(f'./SplitData/S_D_{drop}_{attb}_{nweeks}_Train.csv')

    x = df[df.columns[4:-1]]
    y = df[df.columns[-1]]

    lm = linear_model.LinearRegression()
    model = lm.fit(x, y)

    if lm.score(x, y) > fsval:
        
        with open(f'../Models/{attb}.pickle', 'wb') as f:
            pickle.dump([model, lm.score(x, y)], f)

        models[attb] = (model, lm.score(x, y))

with open(f'../Models/attbs.pickle', 'wb') as f:
    pickle.dump(list(models.keys()), f)

    
# Test data
test = pd.read_csv(f'./SplitData/S_D_{drop}_{attb}_{nweeks}_Test.csv')
out = test[test.columns[0:4]]

for col in models.keys():
    test = pd.read_csv(f'./SplitData/S_D_{drop}_{attb}_{nweeks}_Test.csv')

    xTest = test[test.columns[4:-1]]
    yTest = test[test.columns[-1]]

    lm = models[col][0]

    yPred = lm.predict(xTest)

    out[f'{col}O'] = yTest 
    out[f'{col}P'] = yPred

out['ObsScore'] = [0 for i in range(len(out))]
out['PredScore'] = [0 for i in range(len(out))]

if r == 'r2':
    for col in models.keys():
        out['ObsScore'] = out['ObsScore'] + ((models[col][1])**2)*out[f'{col}O']
        out['PredScore'] = out['PredScore'] + ((models[col][1])**2)*out[f'{col}P']
else:
    for col in models.keys():
        out['ObsScore'] = out['ObsScore'] + out[f'{col}O']
        out['PredScore'] = out['PredScore'] + out[f'{col}P']

final = pd.DataFrame()
for week, data in out.groupby(['Week']):
    data.index = range(len(data))
    O95 = np.quantile(data['ObsScore'], .95)
    O05 = np.quantile(data['ObsScore'], .05)

    P95 = np.quantile(data['PredScore'], .95)
    P05 = np.quantile(data['PredScore'], .05)

    data['Obs'] = ['Increase' if data.loc[i]['ObsScore'] > O95 else 'Decrease' if data.loc[i]['ObsScore'] < O05 else 'Normal' for i in range(len(data))]
    data['Pred'] = ['Increase' if data.loc[i]['PredScore'] > P95 else 'Decrease' if data.loc[i]['PredScore'] < P05 else 'Normal' for i in range(len(data))]

    final = pd.concat([final, data])

n1 = len(final[final['Pred'] == 'Increase'])
n2 = len(final[final['Pred'] == 'Decrease'])
o1 = len(final[final['Obs'] == 'Increase'])
o2 = len(final[final['Obs'] == 'Decrease'])
    
tp1 = len(final[(final['Pred'] == 'Increase') & (final['Obs'] == final['Pred'])])/n1
tp2 = len(final[(final['Pred'] == 'Decrease') & (final['Obs'] == final['Pred'])])/n2
bfp1 = len(final[(final['Obs'] == 'Decrease') & (final['Pred'] == 'Increase')])/n1
bfp2 = len(final[(final['Obs'] == 'Increase') & (final['Pred'] == 'Decrease')])/n2
s1 = len(final[(final['Obs'] == 'Increase') & (final['Obs'] == final['Pred'])])/o1
s2 = len(final[(final['Obs'] == 'Decrease') & (final['Obs'] == final['Pred'])])/o2





# PART 7: OUTPUT STATISTICS

Statistics = pd.DataFrame({'Index':['True Positive 1', 'True Positive 2', 'Bad False Positive 1', 'Bad False Positive 2'], 
                           'Statistics': [tp1, tp2, bfp1, bfp2]})
Statistics.index = Statistics.Index
Statistics.to_csv('./Statistics.csv')
Statistics.to_csv('../Statistics.csv')

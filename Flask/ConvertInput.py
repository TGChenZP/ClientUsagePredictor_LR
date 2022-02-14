from datetime import date, timedelta
import pandas as pd

def find_monday(date):
    return date-timedelta(days=date.weekday())
    
def find_sunday(date):
    return date+timedelta(days=(6-date.weekday()))

def convert_date(input):
    d = input.split('/')[0]
    m = input.split('/')[1]
    y = input.split('/')[2]
    return date(year = int(y), month = int(m), day = int(d))

def convert_back(input):
    return f"{input.day}/{input.month}/{input.year}"

def find_week(input):
    df = pd.read_csv('../DateMatchWeek.csv')
    date = df['dates']
    weekno = df['weekno']
    index = -1
    for i in range(len(weekno)):
        if date[i] == input:
            index = i
            break
    if index == -1:
        return -1
    else:
        return weekno[index]
    
def find_current_week():
    today = date.today()
    inputt = convert_back(today)
    df = pd.read_csv('../DateMatchWeek.csv')
    dates = df['dates']
    weekno = df['weekno']
    index = -1
    for i in range(len(weekno)):
        if dates[i] == inputt:
            index = i
            break
    if index == -1:
        if convert_back(today-timedelta(days=7)) in list(dates):
            return max(weekno)+1
        else:
            return "Please ensure the programme has been run weekly up to the current week"
    else:
        return weekno[index]
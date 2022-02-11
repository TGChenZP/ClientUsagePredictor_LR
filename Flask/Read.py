import pandas as pd
import os

def read_mapping():
    df = pd.read_excel("../Mapping.xlsx")
    return df

mapping = read_mapping()


def read_predictions():
    df = 0
    if os.path.isfile(f'../Predictions.csv'):
        df = pd.read_csv("../Predictions.csv")
    return df
    
predictions = read_predictions()

def read_past_predictions(week):
    if os.path.isfile(f"../History/Week {week}/Predictions.csv"):
        df = pd.read_csv(f"../History/Week {week}/Predictions.csv")
    else:
        return 0
    return df


def read_stats():
    df = 0
    if os.path.isfile(f'../Statistics.csv'):
        df = pd.read_csv("../Statistics.csv")
    return df

stats = read_stats()

def read_past_stats(week):
    if os.path.isfile(f"../History/Week {week}/Statistics.csv"):
        df = pd.read_csv(f"../History/Week {week}/Statistics.csv")
    else:
        return 0
    return df

def read_past_retro(week):
    if os.path.isfile(f"../History/Week {week}/Retrospective.csv"):
        df = pd.read_csv(f"../History/Week {week}/Retrospective.csv")
    else:
        return 0
    return df

def read_current_accuracy():
    if os.path.isfile(f'../Statistics.csv'):
        df = pd.read_csv("../Statistics.csv")
        TruePos1 = int(df['Statistics'][0]*10000)/100
        TruePos2 = int(df['Statistics'][1]*10000)/100
        return [TruePos1, TruePos2]
    else:
        return ['Please Initialise', 'Please Initialise']
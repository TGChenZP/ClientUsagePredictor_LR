from flask import Flask, render_template, url_for, request, redirect
from Read import mapping, predictions, stats, read_past_stats, read_past_retro, read_past_predictions, read_current_accuracy
import pandas as pd
from ConvertInput import find_monday, find_sunday, convert_date, convert_back, find_week, find_current_week
from datetime import date, timedelta
import os
import subprocess

app = Flask(__name__)


# Home Page
@app.route("/")
def home():
    initialised = 0
    if os.path.isfile('../DateMatchWeek.csv'):
        initialised = 1
    if initialised:
        return render_template("home.html", date = date.today(), TruePos1 = read_current_accuracy()[0], TruePos2 = read_current_accuracy()[1], currweek = find_current_week())
    else:
        return render_template("home.html", date=date.today(), TruePos1 = read_current_accuracy()[0], TruePos2 = read_current_accuracy()[1], currweek= "Please Initialise")


# Running the Programme
@app.route("/initial", methods=["GET", "POST"])
def initial():
    if request.method == 'POST':
        init_date = request.form['init_date']
        subprocess.run(f'python InitialTraining.py {init_date}', shell=True, cwd='../Initial')
        return redirect(url_for('finished_init'))
        
    else:
        initialised = 0
        if os.path.isfile('../DateMatchWeek.csv'):
            initialised = 1
        if initialised:
            return render_template("initial.html", initialised = initialised, date = date.today(), currweek = find_current_week())
        else:
            return render_template("initial.html", initialised = initialised, date = date.today(), currweek= "Please Initialise")

@app.route("/finished_init")
def finished_init():
    return render_template("completed_initial.html")
    
@app.route("/run_weekly", methods=["GET","POST"])
def run_weekly():
    if request.method == 'POST':
        run_date = request.form['run_date']
        monday = convert_back(find_monday(convert_date(run_date))-timedelta(days=7))
        sunday = convert_back(find_sunday(convert_date(run_date))-timedelta(days=7))
        subprocess.run(f'python Predict.py {monday} {sunday}', shell=True, cwd = '..')
        subprocess.run(f'python ContinuousTrain.py {monday} {sunday}', shell=True, cwd = '..')
        if os.path.isfile(f'../History/Week {find_week(monday)-2}/Predictions.csv'):
            subprocess.run(f'python Hindsight.py {find_week(monday)-1}', shell=True, cwd = '..' )

        return redirect(url_for('finished_run'))

    else:
        initialised = 0
        if os.path.isfile('../DateMatchWeek.csv'):
            initialised = 1
        if initialised:
            return render_template("run_weekly.html", date = date.today(), TruePos1 = read_current_accuracy()[0], TruePos2 = read_current_accuracy()[1], currweek = find_current_week())
        else:
            return render_template("run_weekly.html", date=date.today(), TruePos1 = read_current_accuracy()[0], TruePos2 = read_current_accuracy()[1], currweek= "Please Initialise")
    
    
@app.route("/run_prediction", methods=["GET","POST"])
def run_prediction():
    if request.method == 'POST':
        run_date = request.form['run_date']
        monday = convert_back(find_monday(convert_date(run_date))-timedelta(days=7))
        sunday = convert_back(find_sunday(convert_date(run_date))-timedelta(days=7))
        subprocess.run(f'python Predict.py {monday} {sunday}', shell=True, cwd = '..')

        return redirect(url_for('finished_run'))

    else:
        initialised = 0
        if os.path.isfile('../DateMatchWeek.csv'):
            initialised = 1
        if initialised:
            return render_template("run_prediction.html", date = date.today(), TruePos1 = read_current_accuracy()[0], TruePos2 = read_current_accuracy()[1], currweek = find_current_week())
        else:
            return render_template("run_prediction.html", date=date.today(), TruePos1 = read_current_accuracy()[0], TruePos2 = read_current_accuracy()[1], currweek= "Please Initialise")
        

@app.route("/run_cont_train", methods=["GET","POST"])
def run_cont_train():
    if request.method == 'POST':
        run_date = request.form['run_date']
        monday = convert_back(find_monday(convert_date(run_date))-timedelta(days=7))
        sunday = convert_back(find_sunday(convert_date(run_date))-timedelta(days=7))
        subprocess.run(f'python ContinuousTrain.py {monday} {sunday}', shell=True, cwd = '..')

        return redirect(url_for('finished_run'))

    else:
        initialised = 0
        if os.path.isfile('../DateMatchWeek.csv'):
            initialised = 1
        if initialised:
            return render_template("run_cont_train.html", date = date.today(), currweek = find_current_week())
        else:
            return render_template("run_cont_train.html", date = date.today(), currweek= "Please Initialise")

        
@app.route("/run_retro", methods=["GET","POST"])
def run_retro():
    if request.method == 'POST':
        run_date = request.form['run_date']
        monday = convert_back(find_monday(convert_date(run_date))-timedelta(days=7))
        sunday = convert_back(find_sunday(convert_date(run_date))-timedelta(days=7))
        if os.path.isfile(f'../History/Week {find_week(monday)-2}/Predictions.csv'):
            subprocess.run(f'python Hindsight.py {find_week(monday)-1}', shell=True, cwd = '..' )

        return redirect(url_for('finished_run'))

    else:
        initialised = 0
        if os.path.isfile('../DateMatchWeek.csv'):
            initialised = 1
        if initialised:
            return render_template("run_retro.html", date = date.today(), currweek = find_current_week())
        else:
            return render_template("run_retro.html", date = date.today(), currweek= "Please Initialise")

        
@app.route("/finished_run")
def finished_run():
    return render_template("completed_run.html")

# Viewing
@app.route("/view_mapping")
def view_mapping():
    return render_template("view_mapping.html", tables=[mapping.to_html(classes='data')], titles=mapping.columns.values)

@app.route('/view_files')
def view_files():
    initialised = 0
    if os.path.isfile('../DateMatchWeek.csv'):
        initialised = 1
    if initialised:
        return render_template("view_files.html", date = date.today(), currweek = find_current_week())
    else:
        return render_template("view_files.html", date = date.today(), currweek= "Please Initialise")

@app.route("/browse_predictions", methods=["GET","POST"])
def browse_predictions():
    if request.method=='POST':
        date1= request.form['date']
        if date1.isdigit():
            df = read_past_predictions(date1)
            if type(df) != int:
                return render_template("view_predictions.html", tables=[df.to_html(classes='data')], titles=df.columns.values, week = date1)
            else:
                return redirect(url_for("error"))
        else:
            df = read_past_predictions(find_week(date1))
            if type(df) != int:
                return render_template("view_predictions.html", tables=[df.to_html(classes='data')], titles=df.columns.values, week = find_week(date1))
            else:
                return redirect(url_for("error"))
        
    else:
        initialised = 0
        if os.path.isfile('../DateMatchWeek.csv'):
            initialised = 1
        if initialised:
            return render_template("browse_predictions.html", date = date.today(), currweek = find_current_week())
        else:
            return render_template("browse_predictions.html", date = date.today(), currweek= "Please Initialise")

@app.route("/view_recent_predictions")
def view_recent_predictions():
    return render_template("view_predictions.html", tables=[predictions.to_html(classes='data')], titles=predictions.columns.values, week = 'Most Recent')

@app.route("/view_recent_statistics")
def view_recent_statistics():
    return render_template("view_statistics.html", tables=[stats.to_html(classes='data')], titles=stats.columns.values, week = 'Most Recent')

@app.route("/browse_statistics", methods=["GET","POST"])
def browse_statistics():
    if request.method=='POST':
        date1 = request.form['date']
        if date1.isdigit():
            df = read_past_predictions(date1)
            if type(df) != int:
                return render_template("view_statistics.html", tables=[df.to_html(classes='data')], titles=df.columns.values, week = date1)
            else:
                return redirect(url_for("error"))
        else:
            df = read_past_predictions(find_week(date1))
            if type(df) != int:
                return render_template("view_statistics.html", tables=[df.to_html(classes='data')], titles=df.columns.values, week = find_week(date1))
            else:
                return redirect(url_for("error"))
        
    else:
        initialised = 0
        if os.path.isfile('../DateMatchWeek.csv'):
            initialised = 1
        if initialised:
            return render_template("browse_statistics.html", date = date.today(), currweek = find_current_week())
        else:
            return render_template("browse_statistics.html", date = date.today(), currweek= "Please Initialise")

@app.route("/browse_retro", methods = ['GET', 'POST'])
def browse_retro():
    if request.method=='POST':
        date1 = request.form['date']
        if date1.isdigit():
            df = read_past_retro(date1)
            if type(df) != int:
                return render_template("view_retro.html", tables=[df.to_html(classes='data')], titles=df.columns.values, week = date1)
            else:
                return redirect(url_for("error"))
        else:
            df = read_past_retro(find_week(date1))
            if type(df) != int:
                return render_template("view_retro.html", tables=[df.to_html(classes='data')], titles=df.columns.values, week = find_week(date1))
            else:
                return redirect(url_for("error"))
        
    else:
        initialised = 0
        if os.path.isfile('../DateMatchWeek.csv'):
            initialised = 1
        if initialised:
            return render_template("browse_retro.html", date = date.today(), currweek = find_current_week())
        else:
            return render_template("browse_retro.html", date = date.today(), currweek= "Please Initialise")
    
@app.route("/error")
def error():
    return render_template('error.html')
import requests
from flask import Blueprint, render_template, request
from flask_login import login_required, current_user
from flask import Flask
from . import db


main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/profile')
@login_required
def profile():
    return render_template('profile.html', name=current_user.name)

@main.route('/main.predict')
@login_required
def predict():
     response = requests.get(url="http://localhost:5000/predict_api")
     return render_template('predict.html')

@main.route('/main.recommendation')
@login_required
def recommendation():
    return render_template('hospital_recommendation.html', name=current_user.name)
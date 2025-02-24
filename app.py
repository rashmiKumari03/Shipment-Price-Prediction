from flask import Flask, render_template, request, jsonify
import warnings
import os

# Ignore all warnings
warnings.filterwarnings("ignore")

# Initialize Flask app
app = Flask(__name__, template_folder='templates')

# Ensure the templates folder exists
if not os.path.exists('templates'):
    raise FileNotFoundError("Templates folder not found. Ensure it is in the correct location.")

# Home route
@app.route('/')
def home():
    return render_template('Home_page.html')

# Price Prediction
@app.route('/prediction')
def price_prediction():
    return render_template('price_prediction.html')

# Dashboard route
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard_page.html')

# KPI route
@app.route('/kpi')
def kpi():
    return render_template('kpi_page.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)

from flask import Flask,render_template,request,redirect
import pickle
import numpy as np
import base64
import pandas as pd
app = Flask(__name__)
model = pickle.load(open('pipe.pkl', 'rb'))
df = pd.read_csv('clean_csv.csv')
@app.route('/',methods=['GET','POST'])
def home():
    companies=list(df['Company'].unique())
    types=sorted(df['TypeName'].unique())
    cpus=sorted(df['Cpu manufacturer '].unique())
    gpus=sorted(df['Gpu manufacturer'].unique())
    oss=sorted(df['OS'].unique())
    weight=sorted(df['Weight'].unique())
    return render_template('index.html', companies=companies, types=types, cpus=cpus, gpus=gpus, oss=oss,weight=weight)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    company = request.form['company']
    laptop_type = request.form['laptop_type']
    cpu = request.form['cpu']
    hdd = int(request.form['hdd'])
    ssd = int(request.form['ssd'])
    screen_size = float(request.form['screen_size'])
    ram = int(request.form['ram'])
    weight = float(request.form['weight'])
    touchscreen = 1 if request.form['touchscreen'] == 'Yes' else 0
    ips = 1 if request.form['ips'] == 'Yes' else 0
    resolution = request.form['resolution']
    gpu = request.form['gpu']
    os = request.form['os']

    # Calculate PPI
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size

    # Create the query array
    query = np.array([company, laptop_type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
    query = query.reshape(1, 12)

    # Perform the prediction
    predicted_price = np.exp(model.predict(query)[0])

    # Return the predicted price as a response
    return render_template('index.html', companies=list(df['Company'].unique()), types=list(df['TypeName'].unique()), cpus=list(df['Cpu manufacturer '].unique()), gpus=list(df['Gpu manufacturer'].unique()), oss=list(df['OS'].unique()), predicted_price=predicted_price, cpu_brand=cpu)

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('iris.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict', methods = ['POST','GET'])
def home():
    try:
        data1 = float(request.form['a'])
        data2 = float(request.form['b'])
        data3 = float(request.form['c'])
        data4 = float(request.form['d'])
        
        arr = np.array([[data1, data2, data3, data4]])
        
        pred = model.predict(arr)[0]
        
        return render_template('predict.html', data=pred)
    except ValueError:
        return "Please provide valid numeric inputs."

if __name__ == '__main__':
    app.run(debug=True)
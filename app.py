import numpy as np
from flask import Flask, request, render_template
import pickle

#Create an app object using the Flask class. 
app = Flask(__name__)

#Load the trained model. (Pickle file)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    features = [np.array(int_features)]
    prediction = model.predict(features)
    
    output = prediction[0]
    
    if output == 0:
        prediction_text = 'Iris-setosa'
    elif output == 1:
        prediction_text = 'Iris-versicolor'
    else:
        prediction_text = 'Iris-virginica'
    
    return render_template('index.html', prediction_text=prediction_text)


if __name__ == "__main__":
    app.run()
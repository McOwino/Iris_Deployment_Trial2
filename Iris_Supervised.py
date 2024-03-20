#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
import pandas as pd
import numpy as np


# modeling libraries
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score


# # Data Preprocessing 

# In[2]:


df = pd.read_csv('iris.csv')

df.head()


# In[3]:


df.drop('Id', axis=1, inplace=True)


# In[4]:


# convert species to numerical
mapping = {'Iris-setosa':0 , 'Iris-versicolor':1,'Iris-virginica':2}

df['Species'].replace(mapping, inplace=True)


# In[5]:


df.head()


# # Modelling

# In[6]:


# instatiate x and y
x=df.drop('Species', axis=1)

y=df['Species']


# In[7]:


# split data into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=400)


# In[8]:


# function for model training
def model_trainer(data, models):
    x_train, x_test, y_train, y_test = data
    for model in models:
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        accuracy = accuracy_score(y_test, preds)
        print(f'Model: {model}, Accuracy: {accuracy}')
    


# In[9]:


# define data
data = (x_train, x_test, y_train, y_test)

# define modeles
dummy = DummyClassifier()
logistic = LogisticRegression(max_iter=10000)
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
svc = SVC()

models = [dummy, logistic, dt, rf, svc]


# In[10]:


# Train models and test best perfommance
model_trainer(data=data, models=models)


# In[11]:


import pickle

# saving the model in a pickle file
model = pickle.dump(rf, open("model.pkl","wb"))

# loading the model int the console
model = pickle.load(open('model.pkl','rb'))


# In[12]:


df.head()


# In[13]:


print(model.predict([[4.7, 3.2, 1.3, 0.2]]))


# In[14]:


df.columns


# In[ ]:


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


# In[ ]:





import csv
import pandas as pd
import numpy as np
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

"""Read the dataset

"""

# Read Raw Dataset
df = pd.read_excel('/Users/shradhasrivastava/flask_auth_app/project/raw_data.xlsx')
data = df.fillna(method='ffill')

"""Data preprocessing"""

def process_data(data):
    data_list = []
    data_name = data.replace('^','_').split('_')
    n = 1
    for names in data_name:
        if (n % 2 == 0):
            data_list.append(names)
        n += 1
    return data_list

disease_list = []
disease_symptom_dict = defaultdict(list)
disease_symptom_count = {}
count = 0

for idx, row in data.iterrows():
    
    # Get the Disease Names
    if (row['Disease'] !="\xc2\xa0") and (row['Disease'] != ""):
        disease = row['Disease']
        disease_list = process_data(data=disease)
        count = row['Count of Disease Occurrence']

    # Get the Symptoms Corresponding to Diseases
    if (row['Symptom'] !="\xc2\xa0") and (row['Symptom'] != ""):
        symptom = row['Symptom']
        symptom_list = process_data(data=symptom)
        for d in disease_list:
            for s in symptom_list:
                disease_symptom_dict[d].append(s)
            disease_symptom_count[d] = count

disease_symptom_dict

disease_symptom_count

f = open('cleaned_data.csv', 'w')

with f:
    writer = csv.writer(f)
    for key, val in disease_symptom_dict.items():
        for i in range(len(val)):
            writer.writerow([key, val[i], disease_symptom_count[key]])

df = pd.read_csv('cleaned_data.csv')
df.columns = ['disease', 'symptom', 'occurence_count']
df.head()

df.replace(float('nan'), np.nan, inplace=True)
df.dropna(inplace=True)

from sklearn import preprocessing

n_unique = len(df['symptom'].unique())
n_unique

df.dtypes

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(df['symptom'])
print(integer_encoded)

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)

onehot_encoded[0]

len(onehot_encoded[0])
cols = np.asarray(df['symptom'].unique())
cols

df_ohe = pd.DataFrame(columns = cols)
print(df_ohe.head())
for i in range(len(onehot_encoded)):
    df_ohe.loc[i] = onehot_encoded[i]

df_ohe.head()

len(df_ohe)

df_disease = df['disease']
df_disease.head()

df_concat = pd.concat([df_disease,df_ohe], axis=1)
df_concat.head()

df_concat.drop_duplicates(keep='first',inplace=True)

df_concat.head()

df_concat.drop_duplicates(keep='first',inplace=True)

df_concat.head()

len(df_concat)

cols = df_concat.columns
cols

cols = cols[1:]

df_concat = df_concat.groupby('disease').sum()
df_concat = df_concat.reset_index()
df_concat[:5]

len(df_concat)

df_concat.to_csv("/Users/shradhasrivastava/flask_auth_app/project/training_data.csv", index=False)

X = df_concat[cols]

# Labels
y = df_concat['disease']

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

len(X_train), len(y_train)

len(X_test), len(y_test)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(X)

dt = DecisionTreeClassifier()

clf_dt=dt.fit(X, y)

"""Accuracy score

"""

clf_dt.score(X, y)

export_graphviz(dt, 
                out_file='./tree.dot', 
                feature_names=cols)



#from graphviz import Source
#from sklearn import tree

#graph = Source(export_graphviz(dt, 
#               out_file=None, 
#               feature_names=cols))

#png_bytes = graph.pipe(format='png')

#with open('tree.png','wb') as f:
#    f.write(png_bytes)

#from IPython.display import Image
#Image(png_bytes)

disease_pred = clf_dt.predict(X)

disease_real = y.values

for i in range(0, len(disease_real)):
    if disease_pred[i]!=disease_real[i]:
        print ('Pred: {0}\nActual: {1}\n'.format(disease_pred[i], disease_real[i]))

import pickle
pickle.dump(clf_dt, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))
print(model)


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler
from flask import Blueprint
from . import db

app1 = Blueprint('app1', __name__)
app = Flask(__name__ , template_folder='/Users/shradhasrivastava/flask_auth_app/project/templates')
model = pickle.load(open('model.pkl', 'rb'))

#@app.route('/')
#def home():
#   return render_template('predict.html')


@app1.route('/predict_api',methods=['POST'])
def predict():

    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
   # scaler = StandardScaler()
    #final_features = scaler.transform(final_features)
    prediction = model.predict(final_features)
    print("final features",final_features)
    print("prediction:",prediction)
    output = prediction[0]
    print(output)


    return render_template('predict.html',prediction_text = 'Predicted Disease: ' + output)
   
   
        

if __name__ == "__main__":
    app.run(host='127.0.0.1', debug=False,port = 5000)


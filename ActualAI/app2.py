from flask import Flask, flash, redirect, render_template, request, session, abort
from  flaskext.mysql import   MySQL
from  flask_uploads   import  UploadSet, configure_uploads, IMAGES
from  random import randint
import json
import pandas as pd
import numpy as np
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
import pickle

app=Flask(__name__,static_url_path="/static")
app.secret_key='ironman'
#api=Api(app)
mysql=MySQL()
def list_to_dict(li): 
     ctt=0 
     dct = {}  
     for item in li:
         
         dct[ctt]=item 
         ctt=ctt+1   
     return dct  
def predict(X_train, y_train, x_test, k):
    # create list for distances and targets
      distances = []
      targets = []

      for i in range(len(X_train)):
        # first we compute the euclidean distance
        distance = np.sqrt(np.sum(np.square(x_test - X_train.values[i, :])))
        # add it to list of distances
        distances.append([distance, i])

    # sort the list
      distances = sorted(distances)
   
    # make a list of the k neighbors' targets
      i=0
      while len(list(set(targets)))<k:
        index=distances[i][1]
        val=y_train[index]
        i=i+1
        targets.append(val)
  
      return list(set(targets))

@app.route('/GetColleges', methods=['POST'])
def Get_Colleges():
    if request.method=='POST':
        gpa=request.form['gpa']
        gre=request.form['gre']
        lang=request.form['lang']
   
    gpa=int(float(gpa))
    gre=int(float(gre))
    lang=int(float(lang))

    df = pd.read_csv("gredataset.csv")
    X=df.iloc[:,[1,2,3]]
    labels=df.iloc[:,[0]]

    from sklearn.preprocessing import LabelEncoder
    le_X=LabelEncoder()
    labels.values[:,0]=le_X.fit_transform(labels.values[:,0])




    labelVal=labels.values.ravel()
    labelVal=labelVal.astype('int')

   
    list=[]
    Xt=([gre,lang,gpa])
    fin=predict(X,labelVal,Xt,8)
   # ovo=OneVsOneClassifier(LinearSVC(random_state=0)).fit(X, labelVal)
    filename = 'finalized_model.sav'
    ovo = pickle.load(open(filename, 'rb'))
    
    perfect=ovo.predict([[gre,lang,gpa]])
    perfect=perfect[0]
    fin.remove(perfect)
  
   
    for op in fin:
      i=labels.index[labels['name'] == op].tolist()
      i=i[0]
      list.append(df.iloc[i,0])
   
    i=labels.index[labels['name'] == perfect].tolist()
    i=i[0]  
    perfect=df.iloc[i,0]

    dict=list_to_dict(list)
    dict.update({'perfect':df.iloc[i,0]})
    return json.dumps(dict, ensure_ascii=False)
  #  return render_template("answer.html",**locals())

@app.route('/GetColleges2', methods=['POST'])
def Get_Colleges():
    if request.method=='POST':
        gpa=request.form['gpa']
        gre=request.form['gre']
        lang=request.form['lang']
        exp=request.form['exp']
   
    gpa=int(float(gpa))
    gre=int(float(gre))
    lang=int(float(lang))
    exp=int(float(exp))

    df = pd.read_csv("gredataset2.csv")
    X=df.iloc[:,[1,2,3,4]]
    labels=df.iloc[:,[0]]

    from sklearn.preprocessing import LabelEncoder
    le_X=LabelEncoder()
    labels.values[:,0]=le_X.fit_transform(labels.values[:,0])




    labelVal=labels.values.ravel()
    labelVal=labelVal.astype('int')

   
    list=[]
    Xt=([gre,lang,gpa])
    fin=predict(X,labelVal,Xt,8)
   # ovo=OneVsOneClassifier(LinearSVC(random_state=0)).fit(X, labelVal)
    filename = 'finalized_model2.sav'
    ovo = pickle.load(open(filename, 'rb'))
    
    perfect=ovo.predict([[gre,lang,gpa]])
    perfect=perfect[0]
    fin.remove(perfect)
  
   
    for op in fin:
      i=labels.index[labels['name'] == op].tolist()
      i=i[0]
      list.append(df.iloc[i,0])
   
    i=labels.index[labels['name'] == perfect].tolist()
    i=i[0]  
    perfect=df.iloc[i,0]

    dict=list_to_dict(list)
    dict.update({'perfect':df.iloc[i,0]})
    return json.dumps(dict, ensure_ascii=False)
  #  return render_template("answer.html",**locals())




if __name__=='__main__':
    app.run(host='0.0.0.0',port=80,debug=True) 

    <h1>{{perfect}}</h1>
	{% for i in list %}
	<br>
	{{i}}
	{% endfor %}
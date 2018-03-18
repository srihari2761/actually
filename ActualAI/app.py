
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
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

app=Flask(__name__,static_url_path="/static")
app.secret_key='ironman'
#api=Api(app)
mysql=MySQL()


app.config['MYSQL_DATABASE_USER']='root'
app.config['MYSQL_DATABASE_PASSWORD']=''
app.config['MYSQL_DATABASE_DB']='aidb'
app.config['MYSQL_DATABASE_HOST']='localhost'
mysql.init_app(app)




def list_to_dict(li):
     ctt=0
     dct = {}
     for item in li:

         dct[ctt]=item
         ctt=ctt+1
     return dct
def GetDetails(clg):
  conn=mysql.connect()
  cursor=conn.cursor()
  print("**"+clg+"***")
  check_stmt=("SELECT * FROM collegetable WHERE collegeName=%s")
  check_data=(clg)

  cursor.execute(check_stmt,check_data)
  #cursor.execute(check_stmt)
  data=cursor.fetchall()
  print(data)
  print(list(data))
  return list(data)     
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

@app.route('/Sopinp')
def Sopinp():
  return render_template('/SOPinp.html')
@app.route('/SOP',methods=['POST'])
def SOP():
  print("working??")
  if request.method=='POST':
    input=request.form['input']
  list_sentences_train=np.load('123.npy')
  print("THIS IS INPUT"+input)
  list_sentences_test=np.array([input])
  list_sentences_test.size
  maxlen=100
  max_features = 15352 # how many unique words to use (i.e num rows in embedding vector)
  tokenizer = Tokenizer(num_words=max_features)
  tokenizer.fit_on_texts(list(list_sentences_train))
  list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
  list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
  X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)
  model = load_model('my_model.h5')
  y_test = model.predict([X_te], batch_size=1024, verbose=1)
  y_classes = y_test.argmax(axis=-1)
  op=y_classes[0]

  return render_template("Sop.html",**locals())



@app.route("/")
def home():
    return render_template("index1.html")
def idgen():
    try:
            conn=mysql.connect()
            cursor=conn.cursor()
    except Exception as e:
            return{'error':str(e)}

    id=randint(0, 1000)
    select_stmt=("SELECT * FROM patient WHERE id=%s")
    select_data=(id)
            #cursor.callproc('spCreatePatient',(_userName,_userPassword,111,_userSex,_userAge,_userAddress,_userPhone,_userWard,_userDate))
            #cursor.execute(    "INSERT INTO patient VALUES ('"+'1'+"','"+_user))
    cursor.execute(select_stmt,select_data)
            #data=cursor.fetchone()
    data=cursor.fetchall()
    if len(data) is 0:
            return id
    else:
            return idgen()


@app.route('/LogIn',methods=['POST'])
def Log_In():
  if request.method=='POST':
    email=request.form['email']
    password=request.form['password']
  conn=mysql.connect()
  cursor=conn.cursor()
  view_stmt=("SELECT * FROM users WHERE Password=%s AND Email=%s")
  view_data=(password,email)
  cursor.execute(view_stmt,view_data)
  data=cursor.fetchall()
  if len(data) is 0:
      pid=0
      message="login failed! try again"
      return render_template('login_fail.html',**locals())
  else:
      return render_template('run.html',**locals())

@app.route('/SignUp',methods=['POST'])
def Sign_Up():
  if request.method=='POST':
    email=request.form['email']
    password=request.form['password']
  conn=mysql.connect()
  cursor=conn.cursor()
  check_stmt=("SELECT * FROM users WHERE email=%s")
  check_data=(email)
  cursor.execute(check_stmt,check_data)
  checkd=cursor.fetchall()
  if len(checkd)>0:
        return render_template('login_fail.html',message='name already exists')
  else:
            insert_stmt=("INSERT INTO users VALUES (%s,%s)")
            insert_data=(email,password)

            cursor.execute(insert_stmt,insert_data)

            data=cursor.fetchall()
            if len(data) is 0:
                conn.commit()
                return render_template('run.html',**locals())
            else:
                message="login failed! try again"
                return render_template('login_fail.html',**locals())



@app.route('/GetColleges', methods=['POST'])
def Get_Colleges():
    if request.method=='POST':
        gpa=request.form['gpa']
        gre=request.form['gre']
        lang=request.form['lang']
    c=0
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
    fin=predict(X,labelVal,Xt,7)
   # ovo=OneVsOneClassifier(LinearSVC(random_state=0)).fit(X, labelVal)
    filename = 'finalized_model.sav'
    ovo = pickle.load(open(filename, 'rb'))

    perfect=ovo.predict([[gre,lang,gpa]])
    perfect=perfect[0]
    if perfect in fin:
      fin.remove(perfect)


    for op in fin:
      i=labels.index[labels['name'] == op].tolist()
      i=i[0]
      list.append(df.iloc[i,0])
    mainList=[]
    for i in list:
      lk=[]
      lk=(GetDetails(i));
      mainList.append(lk);
    #print("mainList"+mainList)
    i=labels.index[labels['name'] == perfect].tolist()
    i=i[0]
    perfect=df.iloc[i,0]
    mainPerfect=GetDetails(perfect)

  #  dict=list_to_dict(list)
  #  dict.update({'perfect':df.iloc[i,0]})
  #  return json.dumps(dict, ensure_ascii=False)
    return render_template("answer.html",**locals())

@app.route('/GetColleges2', methods=['POST'])
def Get_Colleges2():
    if request.method=='POST':
        gpa=request.form['gpa']
        gre=request.form['gre']
        lang=request.form['lang']
        exp=request.form['exp']

    gpa=int(float(gpa))
    gre=int(float(gre))
    lang=int(float(lang))
    exp=int(float(exp))
    exp=exp/100
    df = pd.read_csv("gsvv2.csv")
    X=df.iloc[:,[1,2,3,4]]
    labels=df.iloc[:,[0]]

    from sklearn.preprocessing import LabelEncoder
    le_X=LabelEncoder()
    labels.values[:,0]=le_X.fit_transform(labels.values[:,0])




    labelVal=labels.values.ravel()
    labelVal=labelVal.astype('int')


    list=[]
    Xt=([gre,lang,gpa,exp])
    fin=predict(X,labelVal,Xt,7)
   # ovo=OneVsOneClassifier(LinearSVC(random_state=0)).fit(X, labelVal)
    filename = 'fimo2.sav'
    ovo = pickle.load(open(filename, 'rb'))

    perfect=ovo.predict([[gre,lang,gpa,exp]])
    perfect=perfect[0]
    if perfect in fin:

      fin.remove(perfect)


    for op in fin:
      i=labels.index[labels['name'] == op].tolist()
      i=i[0]
      list.append(df.iloc[i,0])
    mainList=[]
    for i in list:
      mainList.append(GetDetails(i))
          

    i=labels.index[labels['name'] == perfect].tolist()
    i=i[0]
    perfect=df.iloc[i,0]
    mainPerfect=GetDetails(perfect)

  #  dict=list_to_dict(list)
  #  dict.update({'perfect':df.iloc[i,0]})
  #  return json.dumps(dict, ensure_ascii=False)
    list.reverse()
    return render_template("answer.html",**locals())





if __name__=='__main__':
    app.run(host='0.0.0.0',port=80,debug=True)

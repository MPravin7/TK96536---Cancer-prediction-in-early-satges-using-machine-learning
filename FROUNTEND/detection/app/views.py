from django.shortcuts import render, redirect
from django.contrib.auth.models import User 
# Create your views here.
from django.contrib import messages
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from collections.abc import Iterable
from . models import *


def index(request):

    return render(request,'index.html')

def about(request):
    
    return render(request,'about.html')


def login(request):
    if request.method=='POST':
        lemail=request.POST['email']
        lpassword=request.POST['password']

        d=Register.objects.filter(email=lemail,password=lpassword).exists()
        print(d)
        return redirect('userhome')
    else:
        return render(request,'login.html')

def registration(request):
    if request.method=='POST':
        Name = request.POST['Name']
        email=request.POST['email']
        password=request.POST['password']
        conpassword=request.POST['conpassword']
        

        print(Name,email,password,conpassword)
        if password==conpassword:
            rdata=Register(email=email,password=password)
            rdata.save()
            return render(request,'login.html')
        else:
            msg='Register failed!!'
            return render(request,'registration.html')

    return render(request,'registration.html')
    # return render(request,'registration.html')


def userhome(request):
    
    return render(request,'userhome.html')

def load(request):
   if request.method=="POST":
        file=request.FILES['file']
        global df
        df=pd.read_excel(file)
        messages.info(request,"Data Uploaded Successfully")
    
   return render(request,'load.html')

def view(request):
    col=df.to_html
    dummy=df.head(100)
   
    col=dummy.columns
    rows=dummy.values.tolist()
    return render(request, 'view.html',{'col':col,'rows':rows})

    # return render(request,'viewdata.html', {'columns':df.columns.values, 'rows':df.values.tolist()})
    
    
def preprocessing(request):

    global x_train,x_test,y_train,y_test,x,y
    if request.method == "POST":
        # size = request.POST['split']
        size = int(request.POST['split'])
        size = size / 100
        df.drop('Patient Id',axis=1,inplace=True)
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df['Level'] = le.fit_transform(df['Level'])

        x=df.drop('Level',axis=1)
        y=df['Level']

        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=101)

        messages.info(request,"Data Preprocessed and It Splits Succesfully")
   
    return render(request,'preprocessing.html')
 



def model(request):
    if request.method == "POST":

        model = request.POST['algo']

        if model == "0":
            from sklearn.ensemble import RandomForestClassifier
            rf = RandomForestClassifier(n_estimators=52)
            rf = rf.fit(x_train,y_train)
            y_pred = rf.predict(x_test)
            acc_rf=accuracy_score(y_test,y_pred)
            msg = 'Accuracy of RandomForestClassifier : ' + str(acc_rf)
            return render(request,'model.html',{'msg':msg})
        elif model == "1":
            from sklearn.tree import DecisionTreeClassifier 
            dt = DecisionTreeClassifier(criterion="entropy",max_depth=3)
            dt = dt.fit(x_train,y_train)
            y_pred = dt.predict(x_test)
            acc_dt=accuracy_score(y_test,y_pred)
            msg = 'Accuracy of DecisionTreeClassifier :  ' + str(acc_dt)
            return render(request,'model.html',{'msg':msg})
        elif model == "2":
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression()
            lr = lr.fit(x_train,y_train)
            y_pred = lr.predict(x_test)
            acc_lr=accuracy_score(y_test,y_pred)
            msg = 'Accuracy of LogisticRegression :  ' + str(acc_lr)
            return render(request,'model.html',{'msg':msg})     
    return render(request,'model.html')

def prediction(request):

    global x_train,x_test,y_train,y_test,x,y
    

    if request.method == 'POST':

        f1 = float(request.POST['Age'])
        f2 = float(request.POST['Gender'])
        f3 = float(request.POST['Air Pollution'])
        f4 = float(request.POST['Alcohol use'])
        f5 = float(request.POST['Dust Allergy'])
        f6 = float(request.POST['OccuPational Hazards'])
        f7 = float(request.POST['Genetic Risk'])
        f8 = float(request.POST['chronic Lung Disease'])
        f9 = float(request.POST['Balanced Diet'])
        f10 = float(request.POST['Obesity'])
        f11 = float(request.POST['Smoking'])
        f12 = float(request.POST['Passive Smoker'])
        f13 = float(request.POST['Chest Pain'])
        f14 = float(request.POST['Coughing of Blood'])
        f15 = float(request.POST['Fatigue'])
        f16 = float(request.POST['Weight Loss'])
        f17 = float(request.POST['Shortness of Breath'])
        f18 = float(request.POST['Wheezing'])
        f19 = float(request.POST['Swallowing Difficulty'])
        f20 = float(request.POST['Clubbing of Finger Nails'])
        f21 = float(request.POST['Frequent Cold'])
        f22 = float(request.POST['Dry Cough'])
        f23 = float(request.POST['Snoring'])


        PRED = [[f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23]]
       
        from sklearn.tree import DecisionTreeClassifier 
        model = DecisionTreeClassifier()
        model.fit(x_train,y_train)
        xgp = np.array(model.predict(PRED))

        if xgp==0:
            msg = ' <span style = color:white;>The Cancer level is : <span style = color:green;><b>High</b></span></span>'
        elif xgp==1:
            msg = ' <span style = color:white;>The Cancer level is : <span style = color:red;><b>Low</b></span></span>'
        else :
            msg = ' <span style = color:white;>The Cancer level is : <span style = color:red;><b>Medium</b></span></span>'
        
        return render(request,'prediction.html',{'msg':msg})

    
    return render(request,'prediction.html')
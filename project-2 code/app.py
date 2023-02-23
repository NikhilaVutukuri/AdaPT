from msilib.schema import tables
from flask import Flask,render_template,send_file, request
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
import os
import sys
import math

import csv
import pandas as pd
import numpy as np

#preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler as mms
from sklearn.preprocessing import StandardScaler as ss
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split as tts

from imblearn.over_sampling import SMOTE

#models
from sklearn import linear_model
from sklearn import ensemble
from sklearn import neighbors
from sklearn import svm

#metrics
from sklearn import metrics

#saving model
import joblib

#parameters and flags:
model_parameters = {}
performance_report = {}
model_count=1

### Removing redundant features (like names and ids)

def removeRedundant(df):
    if (len(model_parameters['redundant_features']))>0:
        if (set(model_parameters['redundant_features']).issubset(set(df.columns))):
            df = df.drop(model_parameters['redundant_features'] , axis = 1)
    return df

### Missing values imputation

def missingValueImputation(df):
    if(df.isnull().values.any()):
        #if null values exist, get a list of null columns
        null_cols = df.columns[df.isnull().any()]
        #update performance report
        model_parameters['missing_values']= 'imputed'
        for i in null_cols:
            if (df[i].dtype=='O' or df[i].nunique()<10):
                df[i].fillna(df[i].mode()[0], inplace= True)
            else:
                df[i].fillna(df[i].mean(), inplace = True)
    if not df.isnull().values.any():
        return df

### Encoding

##### Label Encoding

def labelEncodeDfValues(df):
    object_columns = df.select_dtypes(include=['object']).columns
    if(len(object_columns))>0:
        model_parameters['encoding']= 'label_encoding'
        le=LabelEncoder()
        for i in object_columns:
            df[i]=le.fit_transform(df[i])
    return df


### Separate features and targets

def separateTarget(df):
    target = model_parameters['target']
    y= df[target]
    x= df.drop(target ,axis=1)
    return x,y


### Balance the data

def dataBalancing(x,y):
    class1 = y.value_counts()[0]
    class2 = y.value_counts()[1]
    ratio = min(class1, class2)/max(class1, class2)

    if(ratio<0.85):
        model_parameters['data_balancing']= 'smote_oversampling'
        sm = SMOTE(random_state=27)
        x,y = sm.fit_resample(x,y)
        print('After balancing, shape: ', x.shape, ' ', y.shape)
    return x,y

### Shuffle and Split data

def trainTestSplit(x,y):
    #x_train,x_test,y_train,y_test
    data = tts(x,y,random_state=42,test_size=0.08)
    return data

### Normalizing and Scaling Data

##### Max Scaling

def maxScaler(df):
    for column in df.columns:
        df[column] = df[column]  / df[column].abs().max()
    #df.head()
    return df

##### Z-scaler

def zScaler(df):
    for column in df:
        df[column] = (df[column] - df[column].mean()) / df[column].std()    
    #df.head()
    return df

##### Min-Max Scaler

def minMaxScaler(df):
    min_max=mms()
    scaled_df =min_max.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_df,columns=df.columns)
    #scaled_df.head()
    return scaled_df

##### Standard Scaler

def standardScaler(df):
    std_scaler=ss()
    scaled_df= std_scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_df,columns=df.columns)
    #scaled_df.head()
    return scaled_df

##### Power transform

def powerTransform(df):
    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    scaled_df = pt.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_df,columns=df.columns)
    #scaled_df.head()
    return scaled_df

### Machine Learning Models

##### Classification

#knn

def createKNNClassifier(x_train,y_train, x_test, y_test, normalization):
    knn = neighbors.KNeighborsClassifier(n_neighbors= 25, weights= "uniform")
    knn.fit(x_train,y_train)
    predictions = knn.predict(x_test)

    #metrics
    global model_count
    report = {
          'normalization': normalization,
          'algorithm': 'KNN_classifier',
          'accuracy' : metrics.accuracy_score(y_test, predictions),
          'precision': metrics.precision_score(y_test, predictions),
          'recall': metrics.recall_score(y_test, predictions),
          'f1_score': metrics.f1_score(y_test, predictions),
          'auc_roc': metrics.roc_auc_score(y_test, predictions)
    }
    performance_report['model_'+str(model_count)] = report
    joblib.dump(knn, f'model_{model_count}.pkl')
    model_count=model_count+ 1


# Logistic regression

def createLogisticRegression(x_train,y_train, x_test, y_test, normalization):
    logreg = linear_model.LogisticRegression()
    logreg.fit(x_train, y_train)
    predictions = logreg.predict(x_test)

    #metrics
    global model_count
    report = {
          'normalization': normalization,
          'algorithm': 'logistic_regression',
          'accuracy' : metrics.accuracy_score(y_test, predictions),
          'precision': metrics.precision_score(y_test, predictions),
          'recall': metrics.recall_score(y_test, predictions),
          'f1_score': metrics.f1_score(y_test, predictions),
          'auc_roc': metrics.roc_auc_score(y_test, predictions)
    }
    performance_report['model_'+str(model_count)] = report
    joblib.dump(logreg, f'model_{model_count}.pkl')
    model_count=model_count+ 1


# Random forest classifier

def createRandomForestClassifier(x_train,y_train, x_test, y_test, normalization):
    rfc= ensemble.RandomForestClassifier()
    rfc.fit(x_train, y_train)
    predictions = rfc.predict(x_test)

    #metrics
    global model_count
    report = {
          'normalization': normalization,
          'algorithm': 'random_forest_classifier',
          'accuracy' : metrics.accuracy_score(y_test, predictions),
          'precision': metrics.precision_score(y_test, predictions),
          'recall': metrics.recall_score(y_test, predictions),
          'f1_score': metrics.f1_score(y_test, predictions),
          'auc_roc': metrics.roc_auc_score(y_test, predictions)
    }
    performance_report['model_'+str(model_count)] = report
    joblib.dump(rfc, f'model_{model_count}.pkl')
    model_count=model_count+ 1


# svm

def createSVMClassifier(x_train,y_train, x_test, y_test, normalization):
    model = svm.SVC()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    #metrics
    global model_count
    report = {
          'normalization': normalization,
          'algorithm': 'SVM_classifier',
          'accuracy' : metrics.accuracy_score(y_test, predictions),
          'precision': metrics.precision_score(y_test, predictions),
          'recall': metrics.recall_score(y_test, predictions),
          'f1_score': metrics.f1_score(y_test, predictions),
          'auc_roc': metrics.roc_auc_score(y_test, predictions)
    }
    performance_report['model_'+str(model_count)] = report
    joblib.dump(model, f'model_{model_count}.pkl')
    model_count=model_count+ 1


##### Regression

#linear regression

def createLinearRegression(x_train,y_train, x_test, y_test, normalization):
    reg = linear_model.LinearRegression()
    reg.fit(x_train, y_train)
    predictions = reg.predict(x_test)

    #metrics
    global model_count
    report = {
          'normalization': normalization,
          'algorithm': 'linear_regression',
          'r2_score' : metrics.r2_score(y_test, predictions),
          'mean_squared_error': metrics.mean_squared_error(y_test, predictions, squared=False),
          'root_mean_squared_error': math.sqrt(metrics.mean_squared_error(y_test, predictions, squared=False)),
          'mean_absolute_error': metrics.mean_absolute_error(y_test, predictions),
    }
    performance_report['model_'+str(model_count)] = report
    joblib.dump(reg, f'model_{model_count}.pkl')
    model_count=model_count+ 1


#svm svr

def createSVMRegressor(x_train,y_train, x_test, y_test, normalization):
    reg = svm.SVR()
    reg.fit(x_train, y_train)
    predictions = reg.predict(x_test)

    #metrics
    global model_count
    report = {
          'normalization': normalization,
          'algorithm': 'SVM_regressor',
          'r2_score' : metrics.r2_score(y_test, predictions),
          'mean_squared_error': metrics.mean_squared_error(y_test, predictions, squared=False),
          'root_mean_squared_error': math.sqrt(metrics.mean_squared_error(y_test, predictions, squared=False)),
          'mean_absolute_error': metrics.mean_absolute_error(y_test, predictions),
    }
    performance_report['model_'+str(model_count)] = report
    joblib.dump(reg, f'model_{model_count}.pkl')
    model_count=model_count+ 1


# knn regressor

def createKNNRegressor(x_train,y_train, x_test, y_test, normalization):
    reg = neighbors.KNeighborsRegressor(n_neighbors= 5, weights= "uniform")
    reg.fit(x_train, y_train)
    predictions = reg.predict(x_test)

    #metrics
    global model_count
    report = {
          'normalization': normalization,
          'algorithm': 'KNN_regressor',
          'r2_score' : metrics.r2_score(y_test, predictions),
          'mean_squared_error': metrics.mean_squared_error(y_test, predictions, squared=False),
          'root_mean_squared_error': math.sqrt(metrics.mean_squared_error(y_test, predictions, squared=False)),
          'mean_absolute_error': metrics.mean_absolute_error(y_test, predictions),
    }
    performance_report['model_'+str(model_count)] = report
    joblib.dump(reg, f'model_{model_count}.pkl')
    model_count=model_count+ 1


#random forest regressor

def createRandomForestRegressor(x_train,y_train, x_test, y_test, normalization):
    reg = ensemble.RandomForestRegressor()
    reg.fit(x_train, y_train)
    predictions = reg.predict(x_test)

    #metrics
    global model_count
    report = {
          'normalization': normalization,
          'algorithm': 'random_forest_regressor',
          'r2_score' : metrics.r2_score(y_test, predictions),
          'mean_squared_error': metrics.mean_squared_error(y_test, predictions, squared=False),
          'root_mean_squared_error': math.sqrt(metrics.mean_squared_error(y_test, predictions, squared=False)),
          'mean_absolute_error': metrics.mean_absolute_error(y_test, predictions),
    }
    performance_report['model_'+str(model_count)] = report
    joblib.dump(reg, f'model_{model_count}.pkl')
    model_count=model_count+ 1


### Implement Models

def implementClassificationModels(x,y, normalization):

    x_train,x_test,y_train,y_test = trainTestSplit(x,y)

    #models
    createKNNClassifier(x_train, y_train, x_test, y_test, normalization)
    createLogisticRegression(x_train,y_train, x_test, y_test, normalization)
    createRandomForestClassifier(x_train,y_train, x_test, y_test, normalization)
    createSVMClassifier(x_train,y_train, x_test, y_test, normalization)

def implementRegressionModels(x,y,normalization):
    x_train,x_test,y_train,y_test = trainTestSplit(x,y)

    #models
    createKNNRegressor(x_train, y_train, x_test, y_test, normalization)
    createLinearRegression(x_train,y_train, x_test, y_test, normalization)
    createRandomForestRegressor(x_train,y_train, x_test, y_test, normalization)
    createSVMRegressor(x_train,y_train, x_test, y_test, normalization)


############# FLASK APP START ###################
app = Flask(__name__)
#app.config['UPLOAD_FOLDER'] = '/datasets'

@app.route("/")
@app.route("/home")
def home_page():
    for file in os.listdir():
        if (file.endswith(".csv") or file.endswith(".pkl")):
            os.remove(file)
    global model_count
    global model_parameters
    global performance_report
    model_count =1
    model_parameters = {}
    performance_report = {}
    return render_template('index.html')

@app.route('/data',methods=['GET','POST'])
def processing():
    for file in os.listdir():
        if (file.endswith(".csv") or file.endswith(".pkl")):
            os.remove(file)
    global model_count
    global model_parameters
    global performance_report
    model_count =1
    model_parameters = {}
    performance_report = {}
    if request.method == 'POST':
        form=request.form
        #print(str(request.files), file=sys.stderr)
        model_parameters['dataset_name'] = request.files['csvfile'].filename
        model_parameters['type_of_problem'] = form['algo']
        if(form['algo']== 'Regression'):
            model_parameters['metric_preference'] = form['regression-metric']
        else:
            model_parameters['metric_preference'] = form['classification-metric']
        if (len(form.getlist('redundant-features'))!=0):
            model_parameters['redundant_features'] = form.getlist('redundant-features')
        else:
            model_parameters['redundant_features']=[]

        model_parameters['target'] = form['target']

        performance_report['type_of_problem']= model_parameters['type_of_problem']
        
        f = request.files['csvfile']
        f.save(secure_filename(f.filename))
        with open(f.filename) as file:
            df=pd.read_csv(file)
        #return form
        
        #------------------

        ### Implement on a dataset

        #call all these functions

        df = removeRedundant(df)

        df = missingValueImputation(df)
        df = labelEncodeDfValues(df)

        df.to_csv(f"{model_parameters['dataset_name'].split('.')[0]}_clean.csv", index=False)

        x,y = separateTarget(df)


        if(model_parameters['type_of_problem']=='Classification'):
            x,y = dataBalancing(x,y)

        #normalize

        x_max = maxScaler(x)
        x_z = zScaler(x)
        x_min_max = minMaxScaler(x)
        x_std = standardScaler(x)
        x_power = powerTransform(x)


        if(model_parameters['type_of_problem']=='Classification'):

            implementClassificationModels(x_max,y, 'max_scaling')
            implementClassificationModels(x_z,y, 'z_scaling')
            implementClassificationModels(x_min_max,y, 'min_max_scaling')
            implementClassificationModels(x_std,y, 'standard_scaling')
            implementClassificationModels(x_power,y,'power_transform')

            #return performance_report

        else:
            implementRegressionModels(x_max,y, 'max_scaling')
            implementRegressionModels(x_z,y, 'z_scaling')
            implementRegressionModels(x_min_max,y, 'min_max_scaling')
            implementRegressionModels(x_std,y, 'standard_scaling')
            implementRegressionModels(x_power,y,'power_transform')

            #return performance_report

        #metrics 
        metric = model_parameters['metric_preference'].lower()
        metric_values = {}
        regression_metrics_list=['root_mean_squared_error','mean_squared_error','mean_absolute_error']

        for i in performance_report.keys():
            if ('model' in i):
                metric_values[i] = performance_report[i][metric]

        models = []
        for key,val in metric_values.items():
            if metric not in regression_metrics_list:
                if val== max(metric_values.values()):
                    models.append(key)
            else:
                if val== min(metric_values.values()):
                    models.append(key)

            

        result = {}
        for i in models:
            result[i] = performance_report[i]
            
        #return result

        #------------------
        
        return render_template('results.html', result = result, model_parameters = model_parameters, performance_report = performance_report)

#download dataset
@app.route('/get-clean-data',methods=['GET','POST'])
def cleanData():
    return send_file(f"{model_parameters['dataset_name'].split('.')[0]}_clean.csv",as_attachment=True,mimetype='text/csv', cache_timeout=0)

#download model
@app.route('/get-model/<model>', methods=['GET','POST'])
def trainedModel(model):
    return send_file(f"{model}.pkl", as_attachment=True,mimetype='text/csv', cache_timeout=0)

if __name__ == '__main__':
    app.run(debug=True)

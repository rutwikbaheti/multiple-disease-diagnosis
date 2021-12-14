from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
from info import *
from doctor import *

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

test=pd.read_csv("Testing.csv",error_bad_lines=False)
x_test=test.drop('prognosis',axis=1)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method=='POST':
        col=x_test.columns
        inputt = [str(x) for x in request.form.values()]
        b=[0]*114
        for x in range(0,114):
            for y in inputt:
                if(col[x]==y):
                    b[x]=1
        b=np.array(b)
        b=b.reshape(1,114)
        prediction = model.predict(b)
        prediction=prediction[0]
    return render_template('prediction.html', pred="{}".format(prediction),info = info(prediction),cause = cause(prediction))

@app.route('/appointment',methods=['POST','GET'])
def appointment():
    d =request.args.get("d")
    doc1_name, doc2_name, doc3_name = doc_name(d)
    doc1_qualification,doc2_qualification,doc3_qualification = doc_qualification(d)
    doc1_specialization,doc2_specialization,doc3_specialization = doc_specialization(d)
    doc1_exp,doc2_exp,doc3_exp = doc_exp(d)
    doc1_info,doc2_info,doc3_info = doc_info(d)
    doc1_appoinment,doc2_appoinment,doc3_appoinment = doc_appoinment(d)
    return render_template('appointment.html',doc_1_name = doc1_name ,doc_2_name = doc2_name,doc_3_name = doc3_name, doc_1_qualification = doc1_qualification,doc_2_qualification=doc2_qualification, doc_3_qualification=doc3_qualification,doc_1_specialization=doc1_specialization,doc_2_specialization=doc2_specialization,doc_3_specialization=doc3_specialization,doc_1_exp=doc1_exp,doc_2_exp=doc2_exp,doc_3_exp=doc3_exp,doc_1_info=doc1_info,doc_2_info=doc2_info,doc_3_info=doc3_info,doc_1_appoinment=doc1_appoinment,doc_2_appoinment=doc2_appoinment,doc_3_appoinment=doc3_appoinment)

if __name__ == "__main__":
    app.run(debug=True)

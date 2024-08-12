from flask import Flask,request,render_template
import pandas
import numpy as np
import pickle

model= pickle.load(open('model.pkl','rb'))

#flask app
app=Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predic():
    pass


#python main
if __name__ == "__main":
    app.run(debug=True)

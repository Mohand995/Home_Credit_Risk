from utility import  *
import json
from flask import Flask,jsonify,request



app=Flask(__name__)

@app.route("/Predict_Default_Risk",methods=['POST'])
def predict():
    data=request.json
    try:
        sample=data['info']
    except KeyError:
        return  jsonify({'error':'NO text sent'})
    
    result=Inference(sample)
    return  result

if __name__=='__main__':
    app.run(host="0.0.0.0", port=5000)

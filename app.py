from flask import Flask,request,render_template
from flask import Response
import os
from flask_cors import CORS, cross_origin
from PredictionDataValidation import pred_validation
from predictFromModel import Prediction

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)
#UPLOAD_FOLDER = "./Uploaded_files"
app.config['UPLOAD_FOLDER'] = "Uploaded_files"

@app.route("/")
@cross_origin()
def hello_world():
    return render_template("index.html")

@app.route("/predict",methods = ["POST"])
@cross_origin()
def predict():
    if request.method == "POST":
        f = request.files['file']
        path = None
        if f.filename!='':
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
            path = "./Uploaded_files"
        else:
            print("Default file choosen")
            path = "./Prediction_Batch_files"
        pred_val = pred_validation(path)  # object initialization
        pred_val.prediction_validation()  # calling the prediction_validation function
        pred = Prediction(path)  # object initialization
            # predicting for dataset present in database
        path = pred.predictionFromModel()
        return render_template("predict_page.html")

if __name__ == "__main__":
    app.run()
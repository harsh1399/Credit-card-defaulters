from flask import Flask,request,render_template,send_file
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
    try:
        if request.method == "POST":
            f = request.files['file']
            print(f)
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
            #return Response("Prediction File created at %s!!!" % path)
            return render_template("predict_page.html",message = "Prediction file created")
    except ValueError:
        return render_template("predict_page.html",message ="Error Occurred! %s" % ValueError)
    except KeyError:
        return render_template("predict_page.html",message ="Error Occurred! %s" % KeyError)
    except Exception as e:
        return render_template("predict_page.html",message ="Error Occurred! %s" % e)

@app.route("/download")
@cross_origin()
def download_file():
    path = "Prediction_Output_File/Predictions.csv"
    return send_file(path,as_attachment=True)

if __name__ == "__main__":
    app.run()
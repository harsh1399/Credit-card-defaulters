from os import listdir
import pandas
from app_logging.logger import app_logger


class dataTransformPredict:
    def __init__(self):
        self.goodDataPath = "Prediction_Raw_Files_Validated/Good_Raw"
        self.logger = app_logger()

    def replaceMissingWithNull(self):
        log_file = open("PredictionLogs/dataTransformLog.txt", 'a+')
        try:
            onlyfiles = [f for f in listdir(self.goodDataPath)]
            for file in onlyfiles:
                data = pandas.read_csv(self.goodDataPath + "/" + file)

                data.to_csv(self.goodDataPath + "/" + file, index=None, header=True)
                self.logger.log(log_file, " %s: File Transformed successfully!!" % file)
            # log_file.write("Current Date :: %s" %date +"\t" + "Current time:: %s" % current_time + "\t \t" +  + "\n")

        except Exception as e:
            self.logger.log(log_file, "Data Transformation failed because:: %s" % e)
            # log_file.write("Current Date :: %s" %date +"\t" +"Current time:: %s" % current_time + "\t \t" + "Data Transformation failed because:: %s" % e + "\n")
            log_file.close()
            raise e
        log_file.close()


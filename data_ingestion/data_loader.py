import pandas as pd

# used for reading data and creating a dataframe
class data_getter:
    def __init__(self,file_object,logger_object):
        self.training_file = 'Training_FileFromDB/InputFile.csv'
        self.file_object = file_object
        self.logger_object = logger_object

    # returns the dataframe created for the input file
    def get_data(self):
        self.logger_object.log(self.file_object,"start of get_data method of data_getter class")
        try:
            self.data = pd.read_csv(self.training_file)  # reading the data file
            self.logger_object.log(self.file_object,
                                   'Data Load Successful.Exited the get_data method of the Data_Getter class')
            return self.data
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_data method of the Data_Getter class. Exception message: ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Data Load Unsuccessful.Exited the get_data method of the Data_Getter class')
            raise Exception()

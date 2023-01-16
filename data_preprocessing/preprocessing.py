import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import CategoricalImputer
from imblearn.over_sampling import RandomOverSampler
class Preprocessor:
    def __init__(self,file_object,logger_obj):
        self.file_object =file_object
        self.logger_object = logger_obj

    # remove unwanted spaces from pandas dataframe
    def remove_unwanted_space(self,data):
        self.logger_object.log(self.file_object, 'Entered the remove_unwanted_spaces method of the Preprocessor class')

        try:
            df_without_spaces = data.apply(
                lambda x: x.str.strip() if x.dtype == "object" else x)  # drop the labels specified in the columns
            self.logger_object.log(self.file_object,
                                   'Unwanted spaces removal Successful.Exited the remove_unwanted_spaces method of the Preprocessor class')
            return df_without_spaces
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in remove_unwanted_spaces method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'unwanted space removal Unsuccessful. Exited the remove_unwanted_spaces method of the Preprocessor class')
            raise Exception()

    # remove unwanted columns
    def remove_columns(self,data,columns):
        self.logger_object.log(self.file_object, 'Entered the remove_columns method of the Preprocessor class')
        try:
            useful_data = data.drop(labels=columns, axis=1)  # drop the labels specified in the columns
            self.logger_object.log(self.file_object,
                                   'Column removal Successful.Exited the remove_columns method of the Preprocessor class')
            return useful_data
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in remove_columns method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Column removal Unsuccessful. Exited the remove_columns method of the Preprocessor class')
            raise Exception()

    # this method separates feature and label columns
    def separate_label_feature(self,data,label_column_name):
        self.logger_object.log(self.file_object, 'Entered the separate_label_feature method of the Preprocessor class')
        try:
            X = data.drop(labels=label_column_name,
                               axis=1)  # drop the columns specified and separate the feature columns
            Y = data[label_column_name]  # Filter the Label columns
            self.logger_object.log(self.file_object,
                                   'Label Separation Successful. Exited the separate_label_feature method of the Preprocessor class')
            return X, Y
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in separate_label_feature method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Label Separation Unsuccessful. Exited the separate_label_feature method of the Preprocessor class')
            raise Exception()

    #check for null values in data
    def is_null_present(self,data):
        self.logger_object.log(self.file_object, 'Entered the is_null_present method of the Preprocessor class')
        null_present = False
        cols_with_missing_values = []
        cols = data.columns
        try:
            null_counts = data.isna().sum()  # check for the count of null values per column
            for i in range(len(null_counts)):
                if null_counts[i] > 0:
                    null_present = True
                    cols_with_missing_values.append(cols[i])
            if null_present:  # write the logs to see which columns have null values
                dataframe_with_null = pd.DataFrame()
                dataframe_with_null['columns'] = data.columns
                dataframe_with_null['missing values count'] = np.asarray(data.isna().sum())
                dataframe_with_null.to_csv(
                    'preprocessing_data/null_values.csv')  # storing the null column information to file
            self.logger_object.log(self.file_object,
                                   'Finding missing values is a success.Data written to the null values file. Exited the is_null_present method of the Preprocessor class')
            return null_present, cols_with_missing_values
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in is_null_present method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Finding missing values failed. Exited the is_null_present method of the Preprocessor class')
            raise Exception()

    # replace all na values
    def impute_missing_values(self,data,cols_with_missing_values):
        self.logger_object.log(self.file_object, 'Entered the impute_missing_values method of the Preprocessor class')
        try:
            imputer = CategoricalImputer()
            for col in cols_with_missing_values:
                data[col] = imputer.fit_transform(data[col])
            self.logger_object.log(self.file_object,
                                   'Imputing missing values Successful. Exited the impute_missing_values method of the Preprocessor class')
            return data
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in impute_missing_values method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Imputing missing values failed. Exited the impute_missing_values method of the Preprocessor class')
            raise Exception()

    #scale numerical columns using the standard scaler
    def scale_numerical_columns(self,data):
        self.logger_object.log(self.file_object,
                               'Entered the scale_numerical_columns method of the Preprocessor class')
        numeric_attributes = ["LIMIT_BAL","BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6",
                              "PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"]
        try:
            #num_df = data[numeric_attributes]
            #num_df = num_df.to_numpy()
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
            scaled_num_df = pd.DataFrame(data=scaled_data, columns=data.columns)

            self.logger_object.log(self.file_object,
                                   'scaling for numerical values successful. Exited the scale_numerical_columns method of the Preprocessor class')
            return scaled_num_df

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in scale_numerical_columns method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'scaling for numerical columns Failed. Exited the scale_numerical_columns method of the Preprocessor class')
            raise Exception()

    # encode categorical columns to numerical values
    def encode_categorical_columns(self,data):
        self.logger_object.log(self.file_object,
                               'Entered the encode_categorical_columns method of the Preprocessor class')
        try:
            cat_df = data.select_dtypes(include=['object']).copy()
            # Using the dummy encoding to encode the categorical columns to numericsl ones
            for col in cat_df.columns:
                cat_df = pd.get_dummies(cat_df, columns=[col], prefix=[col], drop_first=True)
            self.logger_object.log(self.file_object,
                                   'encoding for categorical values successful. Exited the encode_categorical_columns method of the Preprocessor class')
            return cat_df
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in encode_categorical_columns method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'encoding for categorical columns Failed. Exited the encode_categorical_columns method of the Preprocessor class')
            raise Exception()

    # make imbalanced dataset a balanced one
    def handle_imbalanced_dataset(self,x,y):
        self.logger_object.log(self.file_object,
                               'Entered the handle_imbalanced_dataset method of the Preprocessor class')
        try:
            rdsmple = RandomOverSampler()
            x_sampled, y_sampled = rdsmple.fit_sample(x, y)
            self.logger_object.log(self.file_object,
                                   'dataset balancing successful. Exited the handle_imbalanced_dataset method of the Preprocessor class')
            return x_sampled, y_sampled
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in handle_imbalanced_dataset method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'dataset balancing Failed. Exited the handle_imbalanced_dataset method of the Preprocessor class')
            raise Exception()


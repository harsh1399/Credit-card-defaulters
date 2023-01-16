from sklearn.model_selection import train_test_split
from data_ingestion import data_loader
from data_preprocessing import clustering
from data_preprocessing import preprocessing
from best_model import tuner
from app_logging import logger
from file_operations import file_methods
from trainingDataValidation import train_validation
class trainModel:
    def __init__(self):
        self.logging_class = logger.app_logger()
        self.file_obj = open("TrainingLogs/ModelTrainingLog.txt","a+")

    def trainingModel(self):
        self.logging_class.log(self.file_obj,"Training started")
        try:
            # Getting the data from the source
            data_getter = data_loader.data_getter(self.file_obj, self.logging_class)
            data = data_getter.get_data()
            #preprocessing data
            preprocessor = preprocessing.Preprocessor(self.file_obj, self.logging_class)

            # create separate features and labels
            X, Y = preprocessor.separate_label_feature(data, label_column_name='default payment next month')

            # check if missing values are present in the dataset
            is_null_present, cols_with_missing_values = preprocessor.is_null_present(X)

            # if missing values are there, replace them appropriately.
            if (is_null_present):
                X = preprocessor.impute_missing_values(X, cols_with_missing_values)  # missing value imputation

            """ Applying the clustering approach"""

            kmeans = clustering.Clustering(self.file_obj, self.logging_class)  # object initialization.
            number_of_clusters = kmeans.elbow_plot(X)  # using the elbow plot to find the number of optimum clusters

            # Divide the data into clusters
            X = kmeans.create_clusters(X, number_of_clusters)

            # create a new column in the dataset consisting of the corresponding cluster assignments.
            X['Labels'] = Y

            # getting the unique clusters from our dataset
            list_of_clusters = X['Cluster'].unique()

            """parsing all the clusters and looking for the best ML algorithm to fit on individual cluster"""

            for i in list_of_clusters:
                cluster_data = X[X['Cluster'] == i]  # filter the data for one cluster

                # Prepare the feature and Label columns
                cluster_features = cluster_data.drop(['Labels', 'Cluster'], axis=1)
                cluster_label = cluster_data['Labels']

                # splitting the data into training and test set for each cluster one by one
                x_train, x_test, y_train, y_test = train_test_split(cluster_features, cluster_label, test_size=1 / 3,
                                                                    random_state=355)
                # Proceeding with more data pre-processing steps
                train_x = preprocessor.scale_numerical_columns(x_train)
                test_x = preprocessor.scale_numerical_columns(x_test)

                model_finder = tuner.Model_Finder(self.file_obj, self.logging_class)  # object initialization

                # getting the best model for each of the clusters
                best_model_name, best_model = model_finder.get_best_model(train_x, y_train, test_x, y_test)

                # saving the best model to the directory.
                file_op = file_methods.File_Operations(self.file_obj, self.logging_class)
                save_model = file_op.save_model(best_model, best_model_name + str(i))

            # logging the successful Training
            self.logging_class.log(self.file_obj, 'Successful End of Training')
            self.file_obj.close()

        except Exception as e:
        # logging the unsuccessful Training
            self.logging_class.log(self.file_obj, 'Unsuccessful End of Training')
            self.file_obj.close()
            raise Exception

if __name__ == "__main__":
    path = "./Training_Batch_Files"
    train_valObj = train_validation(path) #object initialization
    train_valObj.train_validation()#calling the training_validation function
    trainModelObj = trainModel() #object initialization
    trainModelObj.trainingModel() #training the model for the files in the table



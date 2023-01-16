from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

class Model_Finder:
    def __init__(self,file_object,logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
        self.gnb = GaussianNB()
        self.xgb = XGBClassifier(objective='binary:logistic')
        self.lr = LogisticRegression()
        self.svm = SVC()
        self.rf = RandomForestClassifier()

    # search for the best parameter values for naive bayes using GridSearchCV
    def get_best_param_for_naive_bayes(self,train_x,train_y):
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_naive_bayes method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            param_grid = {"var_smoothing": [1e-9, 0.1, 0.001, 0.5, 0.05, 0.01, 1e-8, 1e-7, 1e-6, 1e-10, 1e-11]}
            # Creating an object of the Grid Search class
            grid = GridSearchCV(estimator=self.gnb, param_grid=param_grid, cv=3, verbose=3)
            # finding the best parameters
            grid.fit(train_x, train_y)
            # extracting the best parameters
            var_smoothing = grid.best_params_['var_smoothing']
            # creating a new model with the best parameters
            self.gnb = GaussianNB(var_smoothing=var_smoothing)
            # training the mew model`
            self.gnb.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'Naive Bayes best params: ' + str(
                                       grid.best_params_) + '. Exited the get_best_params_for_naive_bayes method of the Model_Finder class')

            return self.gnb
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_naive_bayes method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Naive Bayes Parameter tuning  failed. Exited the get_best_params_for_naive_bayes method of the Model_Finder class')
            raise Exception()

    def get_best_param_for_random_forest(self,train_x,train_y):
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_random_forest method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            param_grid = {"n_estimators": [50,100,150],"max_depth":[3,5,10],"criterion":["gini","entropy"]}
            # Creating an object of the Grid Search class
            grid = GridSearchCV(estimator=self.rf, param_grid=param_grid, cv=3, verbose=3)
            # finding the best parameters
            grid.fit(train_x, train_y)
            # extracting the best parameters
            n_estimators = grid.best_params_['n_estimators']
            max_depth = grid.best_params_["max_depth"]
            criterion = grid.best_params_["criterion"]
            # creating a new model with the best parameters
            self.rf = RandomForestClassifier(n_estimators=n_estimators,max_depth = max_depth,criterion = criterion)
            # training the mew model`
            self.rf.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'Random forest best params: ' + str(
                                       grid.best_params_) + '. Exited the get_best_params_for_random_forest method of the Model_Finder class')

            return self.rf
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_random_forest method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Naive Bayes Parameter tuning  failed. Exited the get_best_params_for_random_forest method of the Model_Finder class')
            raise Exception()

    def get_best_param_for_xgboost(self, train_x, train_y):
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_xgboost method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            param_grid_xgboost = {

                "n_estimators": [50, 100, 130],
                "max_depth": range(3, 11, 1),
                "random_state": [0, 50, 100]

            }
            # Creating an object of the Grid Search class
            grid = GridSearchCV(XGBClassifier(objective='binary:logistic'), param_grid_xgboost, verbose=3,
                                     cv=2)
            # finding the best parameters
            grid.fit(train_x, train_y)

            # extracting the best parameters
            random_state = grid.best_params_['random_state']
            max_depth = grid.best_params_['max_depth']
            n_estimators = grid.best_params_['n_estimators']

            # creating a new model with the best parameters
            self.xgb = XGBClassifier(random_state=random_state, max_depth=max_depth,
                                     n_estimators=n_estimators)
            # training the mew model
            self.xgb.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'XGBoost best params: ' + str(
                                       grid.best_params_) + '. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            return self.xgb
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_xgboost method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'XGBoost Parameter tuning  failed. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            raise Exception()

    def get_best_param_for_logistic_regression(self, train_x, train_y):
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_xgboost method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            param_grid = {
                "max_iter":[100,200,300]
            }
            # Creating an object of the Grid Search class
            grid = GridSearchCV(LogisticRegression(), param_grid, verbose=3,
                                     cv=2)
            # finding the best parameters
            grid.fit(train_x, train_y)
            # extracting the best parameters
            max_iter = grid.best_params_['max_iter']
            # creating a new model with the best parameters
            self.lr = LogisticRegression(max_iter=max_iter)
            # training the mew model
            self.lr.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'Logistic Regression best params: ' + str(
                                       grid.best_params_) + '. Exited the get_best_params_for_logistic_regression method of the Model_Finder class')
            return self.lr
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_logistic_regression method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'logistic regression Parameter tuning  failed. Exited the get_best_params_for_logistic_regression method of the Model_Finder class')
            raise Exception()

    def get_best_model(self,train_x,train_y,test_x,test_y):
        self.logger_object.log(self.file_object,
                               'Entered the get_best_model method of the Model_Finder class')
        # create best model for XGBoost
        try:
            # xgboost = self.get_best_param_for_xgboost(train_x, train_y)
            # prediction_xgboost = xgboost.predict(test_x)  # Predictions using the XGBoost Model
            #
            # if len(test_y.unique()) == 1:  # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
            #     xgboost_score = accuracy_score(test_y, prediction_xgboost)
            #     self.logger_object.log(self.file_object, 'Accuracy for XGBoost:' + str(xgboost_score))  # Log AUC
            # else:
            #     xgboost_score = roc_auc_score(test_y, prediction_xgboost)  # AUC for XGBoost
            #     self.logger_object.log(self.file_object, 'AUC for XGBoost:' + str(xgboost_score))  # Log AUC

            rf = self.get_best_param_for_random_forest(train_x, train_y)
            prediction_randomforest = rf.predict(test_x)  # Predictions using the XGBoost Model

            if len(test_y.unique()) == 1:  # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                rf_score = accuracy_score(test_y, prediction_randomforest)
                self.logger_object.log(self.file_object, 'Accuracy for XGBoost:' + str(rf_score))  # Log AUC
            else:
                rf_score = roc_auc_score(test_y, prediction_randomforest)  # AUC for XGBoost
                self.logger_object.log(self.file_object, 'AUC for XGBoost:' + str(rf_score))  # Log AUC

            # create best model for Random Forest
            naive_bayes = self.get_best_param_for_naive_bayes(train_x, train_y)
            prediction_naive_bayes = naive_bayes.predict(
                test_x)  # prediction using the Random Forest Algorithm

            if len(test_y.unique()) == 1:  # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                naive_bayes_score = accuracy_score(test_y, prediction_naive_bayes)
                self.logger_object.log(self.file_object, 'Accuracy for NB:' + str(naive_bayes_score))
            else:
                naive_bayes_score = roc_auc_score(test_y, prediction_naive_bayes)  # AUC for Random Forest
                self.logger_object.log(self.file_object, 'AUC for RF:' + str(naive_bayes_score))
            print("Random forest score:" ,rf_score)
            print("Naive bayes score: ",naive_bayes_score)
            # comparing the two models
            if naive_bayes_score < rf_score:
                return 'RandomForest', rf
            else:
                return 'NaiveBayes', naive_bayes

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_model method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            raise Exception()


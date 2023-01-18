# Credit Card Defaulters Prediction

### Problem statement - 
Increasing rate of credit card deliquency is one of the major issue for financial institutions. The aim is, based on the previous payment history of the customers, can we classify whether a person is likely to default? If we could achieve this, it will help financial institutions to decide whether to issue a credit card or not to a particular person and also decide its credit limit. 

### Dataset - 
Source - https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients <br>
The dataset contains 30000 entries containing payment history and other details(such as- sex, marital status, education status) of different customers between April 2005 to September 2005.

### Classification methodology - 
I have tried to implement the classification method specified in the below research paper - <br>
reference - https://home.ttic.edu/~shubhendu/Papers/clustering_bagging.pdf  <br>
The paper suggests to divide the data into different clusters and then train separate classification model for every cluster. I have divided the training data using k-means algorithm into 3 different clusters and have tried fitting different models(XGBoost, RandomForest, Naive Bayes) on each cluster.
This method is called as "cluster-then-predict" and it helps in improving the accuracy of classification models. <br>
For prediction, we will find the cluster of all the examples using our trained k-means model and the we will apply corresponding classification model on the data of those clusters.

### Azure deployment - 
The project is deployed on Microsoft Azure as a flask application. <br>
Project link - https://credit-card-defaulter-predict.azurewebsites.net/  <br>
Anyone can upload the data which follows same schema as the original dataset and download the prediction for all the examples.

### How to run the project on local machine - 
The machine must contains Python >=3.9
Execute the following commands to clone the repository and for downloading all the required libraries - <br>
Cloning the repository - <br>
```
git clone https://github.com/harsh1399/Credit-card-defaulters.git
```
All the required libraries are mentioned in requirements.txt file. To download all those libraries - <br>
```
pip install -r requirements.txt
```
To execute the flask application -
```
flask --app app run
```





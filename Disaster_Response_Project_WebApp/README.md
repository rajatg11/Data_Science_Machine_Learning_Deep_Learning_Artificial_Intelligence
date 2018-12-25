# Disaster Response Pipeline Project

### Description:
This project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.
Figure Eight has provided the data set containing real messages that were sent during disaster events. Objective is to create a machine learning pipeline to categorize these events so that we can send the messages to an appropriate disaster relief agency.

### Project Components:
Following is the files structure:

1. data: ETL Pipeline

Loads the messages and categories datasets
Merges the two datasets
Cleans the data
Stores it in a SQLite database

disaster_categories.csv - contains the different categories where message can be mapped to
disaster_messages.csv   - contains the messages
DisasterResponse.db     - database contains the transform data
process_data.py         - contains the code for ETL pipeline

2. models: ML Pipeline

Loads data from the SQLite database
Splits the dataset into training and test sets
Builds a text processing and machine learning pipeline
Trains and tunes a model using GridSearchCV
Outputs results on the test set
Exports the final model as a pickle file

classifier.pkl       - contains the final model
train_classifier.py  - contains the code for ML Pipeline

3. app: Flask Web App

Loads the table from ETL Pipeline
Loads the model/classifier from pickle file
Builds a Web App to receive any text and categories it  

run.py    - contains the code to render the model into web app
templates - master.html and go.html contains the html code for web app


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

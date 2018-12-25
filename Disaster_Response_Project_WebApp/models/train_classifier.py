import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# import data load and save libraries
import pandas as pd
from sqlalchemy import create_engine
import re
import numpy as np
import sqlite3
import pickle

# import modeling libraries - sklearn
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, \
    classification_report, precision_score, recall_score, f1_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

def load_data(database_filepath):
    """Load the data from table and cretaet X, y variables

    Args:
    database_filepath: strings. Contains the path of the database

    Returns:
    X: series. Series containing the messages to train
    y: dataframe. Dataframe containing the target variables
    categories: List of strings. COntains the column/category names to predict
    """
    # load the table
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Disaster_Messages', engine)
    # create the X variable with only messages as column
    X = df['message']
    # create the y variable contains the target variables
    y = df.drop(['id','message','original','genre'], axis=1)
    # fetch the categories to predict
    categories = y.columns
    return X, y, categories

def tokenize(text):
    """create the tokens from the text message

    Args:
    text: strings. String containing the text message

    Returns:
    clean_tokens: List of strings. Contains the list of tokens created
    """
    # split the message text into words
    tokens = word_tokenize(text)
    # initialie the lemmatizer
    lemmatizer = WordNetLemmatizer()
    # create the tokens from the text message
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]

    return clean_tokens

def build_model():
    """Model is build to train and predict

    Args: None

    Returns:
    model_pipeline: pipeline. Pipeline containing the parameters to train and
                    predict the data
    """
    # create the pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(LogisticRegression()))
    ])

    # parameter tuning
    parameters = {
            'clf__estimator__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'vect__ngram_range': ((1, 1), (1, 2))
            }

    #GridSearchCV with pipeline
    model_pipeline = GridSearchCV(pipeline, param_grid=parameters)
    return model_pipeline

def multioutput_classification_report(actual, predicted, col_names):

    """Calculate evaluation metrics for ML model

    Args:
    actual: array. Array containing actual labels.
    predicted: array. Array containing predicted labels.
    col_names: list of strings. List containing names for each of the predicted
    fields.

    Returns:
    metrics_df: dataframe. Dataframe containing the accuracy, precision, recall
    and f1 score for a given set of actual and predicted labels.
    """
    metrics = []

    # Calculate evaluation metrics for each set of labels
    for i in range(len(col_names)):
        accuracy = accuracy_score(actual[:, i], predicted[:, i])
        precision = precision_score(actual[:, i], predicted[:, i], \
                    average='weighted')
        recall = recall_score(actual[:, i], predicted[:, i], average='weighted')
        f1 = f1_score(actual[:, i], predicted[:, i], average='weighted')

        metrics.append([accuracy, precision, recall, f1])

    # Create dataframe containing metrics
    metrics = np.array(metrics)
    metrics_df = pd.DataFrame(data = metrics, index = col_names, \
                columns = ['Accuracy', 'Precision', 'Recall', 'F1'])

    return metrics_df

def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate the model based on Test data and print the accuracy metrics

    Args:
    model: pipeline. Model pipeline trained on messages
    X_test: Series. Series contains the messages to test on models
    Y_test: DataFrame. Dataframe containing the labels to predict
    category_names. List of strings. List of category_names to predict

    Returns: None
    """
    Y_test_pred = model.predict(X_test)
    eval_metrics = multioutput_classification_report(np.array(Y_test), \
                    Y_test_pred, category_names)
    print(eval_metrics)

def save_model(model, model_filepath):
    """Save the trained model as pickle object

    Args:
    model: pipeline. Model pipeline trained on messages
    model_filepath: strings. Path where model needs to be saved

    Returns: None
    """
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    """
    Main function which call other functions to load data, model data,
    metric evaluation and save it for later use

    Args:
    python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

    Returns: None
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

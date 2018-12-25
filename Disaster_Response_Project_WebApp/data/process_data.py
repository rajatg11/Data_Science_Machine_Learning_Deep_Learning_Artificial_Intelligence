#import the libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """"
    Create dataframe from the datasets

    Args:
    messages_filepath: string. Path of the messages dataset
    categories_filepath: string. Path of the categories dataset

    Returns:
    df: dataframe. Dataframe containing the combined data from messages
        and categories datasets
    """
    # Read messages dataset
    messages = pd.read_csv(messages_filepath)
    # Read categories dataset
    categories = pd.read_csv(categories_filepath)
    # Merge messages and categories dataset based on column:id
    df = messages.merge(categories, on=['id'])
    # return the dataframe
    return df

def clean_data(df):
    """"
    Clean the dataset such as
        1) categories column transformed to labels and each label
           has binary classes i.e. either 0 or 1
           i.e. "related-1;request-0;offer-0;aid_related-0;" row has been
           transformed into columns as
           related    request    offer   aid_related
              1          0         0         0
        2) No duplicate data in the DataFrame
        3) child_alone label/column has been dropped because it had only 1 class
           i.e. 0

    Args:
    df: dataframe. Dataframe containing data that needs to be cleaned

    Returns:
    df: dataframe. Dataframe containing clean data for modelling
    """

    #Separate the values in categories column based on ';'
    categories = df['categories'].str.split(';', expand=True)
    # create the different labels/columns based on different categories
    row = categories[:1]
    category_colnames = row.apply(lambda x: x.str.split('-')[0][0])
    categories.columns = category_colnames

    #remove 1 or 0 from the column name suffixes and convert columns to numeric
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    #now drop the categories column from the dataframe
    df = df.drop(['categories'], axis=1)
    # concatenate the newly create columns/labels to the dataframe
    df = pd.concat([df,categories], axis=1)
    #drop any duplicates
    df = df.drop_duplicates()
    # As related column has 3 classes 0, 1, and 2 so replacing 2 with 1
    df['related'] = df['related'].replace({2:1})
    # dropping child_alone column because it has only 1 class i.e. 0
    df = df.drop(['child_alone'], axis=1)

    return df

def save_data(df, database_filename):
    """"
    Saves the dataframe as a table into the database

    Args:
    df: dataframe. Dataframe contains the clean data for modelling
    database_filename: string. Contains path of the database

    Returns: None
    """
    # connect to the database
    engine = create_engine('sqlite:///' + database_filename)
    # save the dataframe into a table
    df.to_sql('Disaster_Messages', engine, index=False)

def main():
    """"
    Main function which call other functions to load, clean and save the data

    Args:
    python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

    Returns: None
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()

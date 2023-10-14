import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    inputs: 
        messages_filepath - filepath for messages csv
        categories_filepath - filepath for categories csv
        
    outputs: df_merged - merged dataframe
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df_merged = pd.merge(messages, categories, on='id')
    
    return df_merged

def clean_data(df_merged):
    """
    inputs: df_merged  - merged dataframe with categories and messages metadata
    outputs: df_cleaned - cleaned merged dataframe
    """

    categories = df_merged['categories'].str.split(pat=';', expand=True)
    
    cat_columns = categories.loc[0].apply(lambda x: x.split('-')[0])
    
    categories.columns = cat_columns
    
    for col in categories:
        categories[col] = pd.to_numeric(categories[col].str[-1])
        categories[col] = categories[col].apply(lambda x: 1 if x > 1 else x)
        
    df_merged = df_merged.drop(['categories'], axis=1)
    
    df_merged = pd.concat([df_merged, categories], axis=1)
    
    df_cleaned = df_merged.drop_duplicates(subset=['message'], keep='first')
    
    return df_cleaned

def save_data(df_clean, database_filename):
    """
    inputs:
        df_clean - the cleaned dataframe from earlier
        database_filename - the flename for where the database is stored
        
    outputs: None
    """
    
    engine = create_engine('sqlite:///' + database_filename)
    df_clean.to_sql('DisasterResponse', engine, index=False, if_exists='replace')
    
def main():
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
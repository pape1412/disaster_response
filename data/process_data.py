# Import libraries
import sys
import argparse
import re
import pandas as pd
from sqlalchemy import create_engine


# Create parser for command line arguments
# Parser object and arguments
parser = argparse.ArgumentParser(description = "Data Pre-Processing for Multilabel Text Classification")
parser.add_argument("messages_dir",
                    help = "directory of messages data")
parser.add_argument("categories_dir",
                    help = "directory of categories data")
parser.add_argument("database_dir",
                    help = "directory of database.db")


def load_data(messages_filepath, categories_filepath):
    """ Loads data from file system
        
        Args: 
            messages_filepath (str): Filepath to message data
            categories_filepath (str): Filepath message categories data
        Returns:
            dataframe: Merge dataframe of messages and categories
    """
    # Load data
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merge datasets
    df = messages.merge(categories, how="left", on="id")
    
    return df


def clean_data(df):
    """ Pre-processing of data for future use
        
        Args: 
            df (dataframe): Dataframe to clean
        Returns:
            dataframe: Cleaned dataframe
    """
    # Create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(";", expand=True)
    
    # Extract a list of new column names for categories
    category_columns = categories.iloc[0].str.replace(r'[^a-z_]','').tolist()
    categories.columns = category_columns
    
    # Iterate through the category columns in df to keep only the numeric values
    for column in categories:
        # Set each value by replacing all non-numeric characters
        categories[column] = categories[column].str.replace(r"[^0-9]","")
        categories[column] = categories[column].astype(str).astype(int)
    
    # Drop the categories column from the df
    df.drop("categories", axis=1, inplace=True)
    
    # Concatenate df and categories data frames.
    df = pd.concat([df, categories], axis=1)
    
    # Check for weird looking categories such as other than binary values or zero variance
    for column in df[category_columns]:
        if len(df[column].unique()) == 1:
             df.drop(column, axis=1, inplace=True)
        else:
            df = df[df[column] <= 1]
            
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filename):
    """ Saves pre-processed data to database table
        
        Args:
            df (dataframe): Cleaned data in dataframe
            database_filepath (str): Filepath of database
        Returns:
            None
    """
    # Create db connection and save file to db
    engine = create_engine("sqlite:///{}".format(database_filename))
    engine.execute("drop table if exists messages")
    df.to_sql("messages", engine, index=False)


def main():
    global args
    args = parser.parse_args()
    if len(sys.argv) == 4:
        print('Loading data from ...\n... Messages: {}\n... Categories: {}'
              .format(args.messages_dir, args.categories_dir))
        df = load_data(args.messages_dir, args.categories_dir)

        print('Cleaning data ...')
        df = clean_data(df)
        
        print('Saving data at ...\n... Database: {}'.format(args.database_dir))
        save_data(df, args.database_dir)
        
        print('Cleaned data saved to database.')
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
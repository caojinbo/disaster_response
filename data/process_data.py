import sys
from sqlalchemy import create_engine
import pandas as pd


def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categroies from CSV files to Pandas df
    :param messages_filepath: str, filepath of messages
    :param categories_filepath: str, filepath of categories
    :return: df: dataset combing messages and categories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Merge messages and categories based on common ID
    df = pd.merge(messages, categories, on='id')

    return df


def clean_data(df):
    """
    Clean and transform data set
    :param df: merged messages and categories df
    :return: cleaned and transformed df
    """
    # Split categories to 36 columns and set the names
    categories = df.categories.str.split(";", expand=True)
    row = categories.iloc[0]
    category_colnames = row.str.split("-").str.get(0).tolist()
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        # convert int 2 to 1
        # categories.loc[categories[column] == 2, column] = 1
        # convert int to bool type
        categories[column] = categories[column].astype(bool)

    # remove old categories column, merge expanded categories
    df.drop(columns=['categories'], inplace=True)
    df = pd.concat([df, categories], axis=1)

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    return df



def save_data(df, database_filename):
    """
    Save df to sqlite3 db
    :param df: cleaned dataset
    :param database_filename: database name
    :return: None
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('InsertTableName', engine, if_exists='replace', index=False)


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
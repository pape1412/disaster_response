# Import libraries
import argparse
import os
import sys

import numpy as np
import pandas as pd
import sqlalchemy
import re


# Imports from  sklearn 
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.externals import joblib

from custom_transformer import CharacterCount, WordCount, StartingVerb, tokenize


# Filter out warnings from sklearn
import warnings
warnings.filterwarnings(module='sklearn*', action='ignore')

# Create parser for command line arguments
# Parser object and arguments
parser = argparse.ArgumentParser(description = "Training of Multilabel Text Classification Model")
parser.add_argument("database_dir",
                    help = "directory of database.db")
parser.add_argument("model_dir",
                    help = "directory to save trained model to")


# Functions
def load_data(database_filepath):
    """ Loads data from database
        
        Args: 
            database_filepath (str): Filepath of database
        Returns:
            series: X, Message text data
            dataframe: Y, Message categories/labels
            list: List of category/label names
    """
    # Load data from database
    engine = sqlalchemy.create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("select * from messages", engine)

    X = df["message"]
    Y = df[df.columns.difference(["id","message","original","genre"])]
    category_names = Y.columns.tolist()
    
    return X, Y, category_names


def build_model():
    """ Builds a classification pipeline for random grid search 
        Args: 
            None
        Returns:
            class: RandomizedSearchCV grid search object
    """
    # Define new pipeline including feature unions
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('nlp', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('char_count', CharacterCount()),
            ('word_count', WordCount()),
            ('starting_verb', StartingVerb())
        ])),
        ('clf', MultiOutputClassifier(SGDClassifier(loss="modified_huber",
                                                    alpha=1e-4,
                                                    max_iter=1000,
                                                    tol=1e-3),
                                      n_jobs=-1))
    ])
    
    # Define parameter grid
    parameters =  {
        'features__nlp__vect__ngram_range': [(1, 1), (1, 2)],
        'features__nlp__vect__max_df': [0.75, 1.0],
        'features__nlp__vect__max_features': [None, 1000, 5000],
        'features__nlp__tfidf__use_idf': [True, False],
        'features__transformer_weights': [
                {'nlp': 1.0,'char_count': 0.0,'word_count': 0.0,'starting_verb': 0.0},
                {'nlp': 1.0,'char_count': 0.5,'word_count': 0.5,'starting_verb': 0.5},
                {'nlp': 1.0,'char_count': 0.75,'word_count': 0.75,'starting_verb': 0.75},
                {'nlp': 1.0,'char_count': 1.0,'word_count': 1.0,'starting_verb': 1.0},
        ]
    }

    # Create cross validation object for randomized grid search
    cv = RandomizedSearchCV(pipeline, param_distributions=parameters, cv=3, n_iter=2)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """ Evaluates classification performance of trained model
        
        Args:
            model (class): Multi-label classifier
            X_test (dataframe): Independet variable(s)
            Y_test (dataframe): Dependent variable(s)
            categoriy_names (list): List of labels
        Returns:
            None
    """
    # Predict on test data
    Y_pred = model.predict(X_test)
    
    # Return results
    print(classification_report(Y_test, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """ Save model to disk
        
        Args: 
            model (class): Multi-label classifier
            model_filepath (str): Path to save model to
        Returns:
            None
    """
    joblib.dump(model, model_filepath, compress = 1)


# Main
def main():
    global args
    args = parser.parse_args()
    
    if len(sys.argv) == 3:
        print('Loading data from ...\n    {}'.format(args.database_dir))
        X, Y, category_names = load_data(args.database_dir)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model for randomized grid search ...')
        model = build_model()
        
        print('Training model with randomized grid search ...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model ...')
        evaluate_model(model.best_estimator_, X_test, Y_test, category_names)

        print('Saving best performing model at ...\n    {}'.format(args.model_dir))
        save_model(model.best_estimator_, args.model_dir)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
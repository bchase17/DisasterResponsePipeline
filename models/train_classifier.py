import sys
import re
import pandas as pd
from sqlalchemy import create_engine
import nltk
import numpy
import os
import pickle
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report

def load_data(database_filepath):
    """
    inputs: database_filepath - path to SQLite destination database
    outputs:
        X - features dataframe
        y - labels dataframe
        category_names - list of category names
    """
    
    engine = create_engine('sqlite:///' + database_filepath)
    df_sql = pd.read_sql_table('DisasterResponse', engine)
    
    X = df_sql['message']
    y = df_sql.iloc[:, 4:]
    category_names = y.columns
    
    return X, y, category_names
    


def tokenize(text):
    """
    Tokenize the text function
    
    inputs: text - text that we are tokenizing
    outputs: clean_tokens - list of tokenized text 
    
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    
    # replace urls with placeholder text
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    # tokenize words from text
    tokens = nltk.word_tokenize(text)
    
    # remove derivatives of each word
    lemmatizer = nltk.WordNetLemmatizer()
    
    # token list
    clean_tokens = [lemmatizer.lemmatize(token).lower().strip() for token in tokens]
    
    return clean_tokens

def build_model():
    """
    inputs: none
    outputs: cv - model containing the model pipeline after conducting a grid search  
    
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    parameters = {
        'clf__estimator__learning_rate':[0.01, 0.1, 0.5]
    }
        
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=2)
    
    return cv
    

def evaluate_model(model, X_test, Y_test, category_names):
    """
    inputs: 
        model - ML model
        X_test - test features for model
        y_test - categories model is testing against
        category_names - category names for y variable
        
    outputs: none
    """
    y_pred=model.predict(X_test)
    print(classification_report(Y_test.values, y_pred, target_names=category_names))
    
def save_model(model, model_filepath):
    """
    inputs:
        model - ML model
        model_filepath: filepath for ML model
        
    outputs: None
    """
    
    pickle.dump(model, open(model_filepath, 'wb'))          

def main():
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
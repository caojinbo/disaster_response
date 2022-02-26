import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

import sys
# import libraries
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import pickle
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, classification_report, accuracy_score, make_scorer, fbeta_score
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    """
    Load data from sqlite database to dataframe
    :param database_filepath: path to sqlite database
    :return: X, Y data for training and testing
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("InsertTableName", engine)
    X = df.message
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns.tolist()
    return X, Y, category_names



def tokenize(text):
    """
    For given text, normalize, tokenize, remove stopwords, and lemmatize it
    :param text: str, text for tokenization
    :return: clean words
    """
    # Normalize and remove punctuations
    text = re.sub(r"[^a-zA-Z0-9]", ' ', text.lower())
    # Tokenize
    words = word_tokenize(text)
    # Remove Stopwords
    words = [w for w in words if w not in stopwords.words('english')]
    # Lemmatize all the nouns and verbs
    lemmatizer = WordNetLemmatizer()
    clean = [lemmatizer.lemmatize(w, pos='n').strip() for w in words]
    clean = [lemmatizer.lemmatize(w, pos='v').strip() for w in clean]

    return clean


def build_model():
    """
    Build a ML pipeline with RandomForest classifier GriSearch
    :return: GridSearch Output
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {'clf__estimator__n_estimators': [50, 60],
                  'clf__estimator__min_samples_split': [2, 3, 4],
                  'clf__estimator__criterion': ['entropy', 'gini']
                 }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv



def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model performance
    :param model: Model to be evaluated
    :param X_test: Test features data
    :param Y_test: Test labels data
    :param category_names: labels of 36 categories
    :return: print accuracy and classification report
    """
    Y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        print("Category:", category_names[i], "\n", classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))
        print('Accuracy of %25s: %.2f' % (category_names[i], accuracy_score(Y_test.iloc[:, i].values, Y_pred[:, i])))


def save_model(model, model_filepath):
    """
    Save model as a pickle file
    :param model: Model to be saved
    :param model_filepath: path of the output pickle file
    :return:
    """
    with open(model_filepath, "wb") as fh:
        pickle.dump(model, fh)


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
import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine
import nltk
nltk.download('stopwords')

app = Flask(__name__)

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

# load data
engine = create_engine('sqlite:///../data/disaster_db.db')
df = pd.read_sql_table('InsertTableName', engine)

# load model
model = joblib.load("../models/disaster_model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # 1) Genre statistics
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # 2) Categories statistics
    cate_counts = df.iloc[:, 4:].sum()
    cate_names = df.iloc[:, 4:].columns
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=cate_names,
                    y=cate_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories",
                    'tickangle': 30
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()
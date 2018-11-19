import json
import plotly
import pandas as pd
import sys
sys.path.append("models")

from custom_transformer import tokenize
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

app = Flask(__name__)

# Load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# Load model
model = joblib.load("models/classifier.pkl")

# Index webpage displays visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # Extract data for visuals
    # Genres
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Categories
    Y = df[df.columns.difference(["id","message","original","genre"])]
    label_counts = Y.sum().values.tolist()
    label_names = Y.sum().index.tolist()
    label_names = [label.replace('_', ' ').title() for label in label_names]
    
    # Words (hard-coded due to performance issues)
    word_counts = [3040, 3008, 2902, 2639, 2495]
    word_labels = ['water', 'people', 'food', 'help', 'need']
    word_labels = [label.replace('_', ' ').title() for label in word_labels]
    
    # Create charts
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    marker=dict(
                        color='rgb(52, 58, 64)'
                    )
                )
            ],
            
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Frequency"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=word_labels,
                    y=word_counts,
                    marker=dict(
                        color='rgb(52, 58, 64)'
                    )
                )
            ],
                    
            'layout': {
                'title': 'Top 5 Words in Message Data',
                'yaxis': {
                    'title': "Frequency"
                },
                'xaxis': {
                    'title': "Word"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=label_names,
                    y=label_counts,
                    marker=dict(
                        color='rgb(52, 58, 64)'
                    )
                )
            ],
            
            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Frequency"
                },
                'xaxis': {
                    'tickangle': -35,
                    'automargin': True
                }
            }
        }
    ]
    
    # Encode plotly graphs as JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# Web page that handles user query and displays model results
@app.route('/go')
def go():
    # Save user input in query
    query = request.args.get('query', '')
    message = ('"' + query + '"' if len(query) > 0 else query)

    # Use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=message,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

    
if __name__ == '__main__':
    main()
import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
import reviews_cossim as rc

# ROOT_PATH for linking with all your files.
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(
    current_directory, 'init.json')

# Assuming your JSON data is stored in a file named 'init.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)
    # authors_df = pd.DataFrame(data['authors'])
    # episodes_df = pd.DataFrame(data['episodes'])
    # reviews_df = pd.DataFrame(data['reviews'])

app = Flask(__name__)
CORS(app)

# Sample search using json with pandas
# query rn is the exact author name


def json_search(query):
    # calculate reviews cossim
    inv_idx = rc.inverted_index(data)
    n_authors = len(data.keys())
    idf = rc.compute_idf(inv_idx, n_authors)
    norms = rc.compute_doc_norms(inv_idx, idf)
    matches_filtered = {}
    query_author_word_counts = rc.author_word_counts(data, query.lower())
    #author/key not in data
    if (len(query_author_word_counts) == 0):
        matches_filtered["first"] = "none"
    else:
        # cossim = list of tuples (score, author name)
        cossim = rc.index_search(query_author_word_counts, inv_idx, idf, norms)
        # if input author has no reviews
        if cossim == []:
            return json.dumps({})
        # filter out top 3 authors, excluding self
        top = cossim[1:4]
        if data[top[0][1]]["book_title"] == []:
            matches_filtered["first"] = (
                round(100*top[0][0], 1), top[0][1], "unavailable")
        else:
            books = list(data[top[0][1]]["book_title"][0].keys())
            matches_filtered["first"] = (
                round(100*top[0][0], 1), top[0][1], books[0])

        if data[top[1][1]]["book_title"] == []:
            matches_filtered["second"] = (
                round(100*top[1][0], 1), top[1][1], "unavailable")
        else:
            books = list(data[top[1][1]]["book_title"][0].keys())
            matches_filtered["second"] = (
                round(100*top[1][0], 1), top[1][1], books[0])

        if data[top[2][1]]["book_title"] == []:
            matches_filtered["third"] = (
                round(100*top[2][0], 1), top[2][1], "unavailable")
        else:
            books = list(data[top[2][1]]["book_title"][0].keys())
            matches_filtered["third"] = (
                round(100*top[2][0], 1), top[2][1], books[0])
    matches_filtered_json = json.dumps(matches_filtered)
    return matches_filtered_json


@app.route("/")
def home():
    return render_template('base.html', title="sample html")


@app.route("/episodes")
def episodes_search():
    text = request.args.get("title")
    return json_search(text)


if 'DB_NAME' not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)

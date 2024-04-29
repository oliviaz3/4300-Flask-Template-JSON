import json
import os
from flask import Flask, render_template, request
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
import reviews_cossim as rc
import numpy as np
import svd as svd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
import re

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

# Sample search using json with pandas



def best_book(author):
    """
    Returns (book title, rating, genre) of the highest rated book for [author]
    """

    titles = []
    ratings = []
    genres = []
    book_ids = []
    for book in data[author]["book_title"]:
        title = list(book.keys())[0]
        titles.append(title)
        ratings.append(book[title]["rating"])
        genres.append(book[title]["genre"])
        book_ids.append(book[title]["id"])
    best_ind = np.argmax(ratings)
    return titles[best_ind], ratings[best_ind], genres[best_ind], book_ids[best_ind]


def get_author_genres(author):
    genres = data[author]["author_genres"]
    # get rid of the last ,
    if genres[-1] == ",":
        genres = genres[:-1]
    # add space after commas to separate genres
    genres = re.sub(r'(?<=[,])(?=[^\s])', r' ', genres)
    return genres


def get_author_index(data, query):
    """
    Returns the index of an author in the original data. If query is not in the
    the author keys return -1.
    """
    authors = list(data.keys())
    auth_ind = -1
    if query in authors:
        auth_ind = authors.index(query)
    return auth_ind

def get_book_website(title, book_id):
    url = "https://www.goodreads.com/book/show/"
    return url + str(book_id) + "." + title

def get_website(author):
    url = "https://www.goodreads.com/author/show/"
    auth_id = data[author]["author_id"]
    return url + str(auth_id) + "." + author

# def combine_author_reviews(data, author1, author2):
#     """
#     Combine the reviews of two authors into a single set of documents.
#     """
#     combined_reviews = []
#     documents = []
#     for name in data:
#         review_text = ""
#         for tok in data[name]["reviews"]:
#             for word in tok:
#                 review_text += (word + " ")
#         documents.append((name, review_text))


# def get_svd_authors(data, query_authors):
#     """
#     Calculates SVD and outputs a list of tuples with author name and score
#     in descending order based on score. 
#     """
#     combined_documents = combine_author_reviews(data, query_authors[0], query_authors[1])
#     print(combined_documents)

#     vectorizer = TfidfVectorizer(max_df=0.7, min_df=1)
#     td_matrix = vectorizer.fit_transform([x[1] for x in combined_documents])
#     docs_compressed, _, _ = svds(td_matrix, k=40)
#     docs_compressed = normalize(docs_compressed)

#     # Get the SVD representation of each author's combined documents
#     author_indices = [get_author_index(
#         data, author.lower()) for author in query_authors]
#     author_reprs = [docs_compressed[author_index]
#                     for author_index in author_indices]

#     if len(author_indices) == 0:
#         return []
#     similarity_scores = []
#     for i, author_repr in enumerate(author_reprs):
#         for j, other_author_repr in enumerate(author_reprs):
#             if i != j:  # Skip self-comparison
#                 similarity_score = np.dot(author_repr, other_author_repr)
#                 similarity_scores.append(
#                     (query_authors[i], query_authors[j], similarity_score))

#     # Sort the similarity scores in descending order
#     similarity_scores.sort(key=lambda x: x[2], reverse=True)

#     return similarity_scores


def get_svd_authors(data, query):
    """
    Calculates svd and outputs a list of tuples with author name and score
    in descending order based on score.
    """
    ind = get_author_index(data, query.lower())
    if ind == -1:
        return []
    else:
        docs = svd.create_docs(data)
        vectorizer = TfidfVectorizer(max_df=.7, min_df=1)
        td_matrix = vectorizer.fit_transform([x[1] for x in docs])
        docs_compressed, s, words_compressed = svds(td_matrix, k=40)
        words_compressed = words_compressed.transpose()
        docs_compressed_normed = normalize(docs_compressed)
        output = svd.closest_author(
        docs, ind, docs_compressed_normed, len(data.keys()))
    return output


def get_cossim_authors(data, query1, query2):
    inv_idx = rc.inverted_index(data)
    n_authors = len(data.keys())
    idf = rc.compute_idf(inv_idx, n_authors)
    norms = rc.compute_doc_norms(inv_idx, idf)
    query_author_word_counts = rc.author_word_counts(data, query1.lower())
    if query2 is not None:
        query_author_word_counts2 = rc.author_word_counts(data, query2.lower())
        query_author_word_counts.update(query_author_word_counts2)

    if (len(query_author_word_counts) == 0):
        return []

    return rc.index_search(query_author_word_counts, inv_idx, idf, norms)


def normalize_sim(score_list):
    """
    Normalize and sort the outputs alphabetically to be aggregated
    """
    if len(score_list) == 0:
        return []
    else:
        divisor = score_list[0][1]
        scores = [(name, score / divisor) for name, score in score_list]

    return scores


def combine_scores(svd, cossim, svd_weight=1, cossim_weight=1):
    """
    Combine the SVD and the cossim similarity scores into one
    """
    # Sort alphabetically
    svd = sorted(svd, key=lambda x: x[0])
    cossim = sorted(cossim, key=lambda x: x[0])

    sum_scores = []
    i, j = 0, 0

    while i < len(svd) and j < len(cossim):
        svd_name = svd[i]
        cossim_name = cossim[j]
        if svd_name[0] == cossim_name[0]:
            sum_score = (svd[i][1]*svd_weight) + (cossim[j][1]*cossim_weight)
            sum_scores.append((svd_name[0], sum_score))
            i += 1
            j += 1
        else:
            j += 1

    result = sorted(sum_scores, key=lambda x: x[1], reverse=True)
    return result


def bins(score):
    score_label = None

    if score <= 0.3333:
        score_label = "Low"
    elif score > 0.3333 and score <= 0.6666:
        score_label = "Medium"
    elif score > 0.6666 and score <= 1:
        score_label = "High"

    return score_label


def edit_distance(str1, str2, m, n):
    dp = [[0 for x in range(n + 1)] for x in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i

    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1])

    return dp[m][n]


@app.route("/search_authors")
def search_authors_route():
    query = request.args.get("query")
    similar_authors = search_authors(query, data.keys())
    return json.dumps(similar_authors)


def search_authors(query, authors):
    filtered_authors = []
    for author in authors:
        distance = edit_distance(
            query.lower(), author.lower(), len(query), len(author))
        if distance <= 20:
            filtered_authors.append({"name": author, "distance": distance})
    filtered_authors.sort(key=lambda x: x["distance"])
    return filtered_authors[:5]


def combine_reviews(authors_data):
    """
    Combine the reviews of books authored by multiple authors into a single metareview.
    """
    combined_reviews = ""
    for author_data in authors_data.values():
        for book in author_data.get("book_title", []):
            combined_reviews += book.get("summary", "")
    return combined_reviews


def json_search(query1, query2):
    authors = data.keys()
    lower_query1 = query1.lower()
    lower_query2 = query2.lower()
    
    matches_filtered = {}

    if (lower_query1 in authors and not lower_query2) or (lower_query1 in authors and lower_query2 in authors):

        if lower_query2:
            cossim_score = get_cossim_authors(data, lower_query1, lower_query2)
        else:
            cossim_score = get_cossim_authors(data, lower_query1, None)
        

        svd_score = normalize_sim(get_svd_authors(
            data, query1))

        if len(cossim_score) == 0 or len(svd_score) == 0:
            matches_filtered[1] = "none"
        else:
            # (name, score)
            combined_scores = normalize_sim(
                combine_scores(cossim_score, svd_score))
            # if input author has no reviews
            if len(combined_scores) == 0:
                return json.dumps({})

            #filter out top 10 authors, excluding query authors
            unfiltered = combined_scores[0:12]
            top = []
            for tup in unfiltered:
                if tup[0] == lower_query1:
                    continue 
                if query2 and tup[0] == lower_query2:
                    continue
                top.append(tup)
            top = top[:10]

            # add in author 1
            query_author = lower_query1
            if data[query_author.lower()]["book_title"] == []:
                matches_filtered["author_1"] = (
                    100,
                    query_author.title(),
                    get_author_genres(query_author.lower()),
                    "unavailable",
                    "unavailable",
                    bins(100),
                    get_website(query_author)
                )
            else:
                book = best_book(query_author)
                # (score, name, genres, book title, book genre, similarity rating, author website, book website)
                matches_filtered["author_1"] = (
                    100,
                    query_author.title(),
                    get_author_genres(query_author),
                    book[0],
                    book[2],
                    100,
                    get_website(query_author),
                    get_book_website(book[0], book[3])

                )
            # add in author 2
            if query2:
                query_author = lower_query2
                if data[query_author.lower()]["book_title"] == []:
                    matches_filtered["author_2"] = (
                        100,
                        query_author.title(),
                        get_author_genres(query_author.lower()),
                        "unavailable",
                        "unavailable",
                        bins(100),
                        get_website(query_author),
                        ""
                    )
                else:
                    book = best_book(query_author)
                    # (score, name, genres, book title, book genre, similarity rating, author website)
                    matches_filtered["author_2"] = (
                        100,
                        query_author.title(),
                        get_author_genres(query_author),
                        book[0],
                        book[2],
                        100,
                        get_website(query_author),
                        get_book_website(book[0], book[3])
                    )

            for idx, tup in enumerate(top):
                if data[tup[0]]["book_title"] == []:
                    # (score, name, genres, book title, book genre, similarity rating, author website)
                    matches_filtered[idx] = (
                        round(100*tup[1], 1), 
                        tup[0].title(),
                        get_author_genres(tup[0]),
                        "unavailable",
                        "unavailable",
                        bins(round(tup[1], 4)),
                        get_website(tup[0]),
                        ""
                    )
                else:
                    book = best_book(tup[0])
                    # (score, name, genres, book title, book genre, similarity rating, author website)
                    matches_filtered[idx] = (
                        round(100*tup[1], 1),
                        tup[0].title(),
                        get_author_genres(tup[0]),
                        book[0],
                        book[2],
                        bins(round(tup[1], 4)),
                        get_website(tup[0]),
                        get_book_website(book[0], book[3])
                    )

    matches_filtered_json = json.dumps(matches_filtered)
    return matches_filtered_json



@app.route("/")
def home():
    return render_template('base.html', title="sample html")


@app.route("/episodes")
def episodes_search():
    authone = request.args.get("authorone")
    authtwo = request.args.get("authortwo")
    return json_search(authone, authtwo)


if 'DB_NAME' not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)

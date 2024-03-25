import json
import os
from collections import Counter
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
import math


def inverted_index(data):
    """
    Returns:
    inverted_index: dict
          inverted_index[term] = list of tuples (author name, term frequency)
    """

    inverted_index = {}
    for author, author_data in data.items():
        reviews = author_data["reviews"]
        # create a term frequency dictionary for the author's reviews
        term_freqs = Counter(reviews)
        # Update the inverted index with author and term frequencies
        for term, freq in term_freqs.items():
            inverted_index.setdefault(term, []).append((author, freq))

    return inverted_index


def compute_idf(inv_idx, n_docs, min_df=5, max_df_ratio=0.95):
    """Compute term IDF values from the inverted index.
    Words that are too frequent or too infrequent get pruned.

    inv_idx: dict
          inverted_index[term] = list of tuples (author name, term frequency) an inverted index as above

    n_docs: int,
        The number of documents.

    min_df: int,
        Minimum number of documents a term must occur in.
        Less frequent words get ignored.
        Documents that appear min_df number of times should be included.

    max_df_ratio: float,
        Maximum ratio of documents a term can occur in.
        More frequent words get ignored.

    Returns
    =======

    idf: dict
        For each term, the dict contains the idf value.

    """
    result = {}
    for word in inv_idx:
        df = len(inv_idx[word])
        if df >= min_df:
            idf = round(math.log(n_docs/(1+df), 2), 4)
            if df/n_docs <= max_df_ratio:
                result[word] = idf
    return result


def compute_doc_norms(inv_idx, idf):
    """Precompute the euclidean norm of each document.

    index: dict
          inverted_index[term] = list of tuples (author name, term frequency) an inverted index as above

    idf: dict,
        Precomputed idf values for the terms.
        idf[word] = idf value.

    Return:
    norms: dict
        norms[author name] = the norm of author
    """
    norms = {}
    for term in inv_idx:
        if term in idf.keys():
            for author_name, word_count in inv_idx[term]:
                sum_of = word_count * idf[term]
                if author_name in norms:
                    norms[author_name] += sum_of**2
                else:
                    norms[author_name] = sum_of**2
    for key in norms:
        norms[key] = math.sqrt(norms[key])

    return norms


def author_word_counts(data, name):
    """
    Return:
    author_word_count: dict
          author_word_count[author name] = {term1: tf1, term2: tf2...}
    """
    author_word_count = {}
    names = list(data.keys())
    if name in names:
        for word in data[name]["reviews"]:
            if word in author_word_count.keys():
                author_word_count[word] += 1
            else:
                author_word_count[word] = 1
    return author_word_count


def accumulate_dot_scores(author_word_counts, inv_idx: dict, idf: dict) -> dict:
    """Perform a term-at-a-time iteration to efficiently compute the numerator term of cosine similarity across multiple documents.

    Arguments
    =========

    author_word_counts: dict,
        A dictionary containing all words that appear in the reviews for the query author;
        Each word is mapped to a count of how many times it appears in the reviews.
        In other words, author_word_counts[w] = the term frequency of w in the reviews.

    inv_idx: the inverted index as above,

    idf: dict,
        Precomputed idf values for the terms.

    doc_scores: dict
        Dictionary mapping from author name to the final accumulated score for that author
    """
    doc_scores = {}

    # Iterate through each term (word) in the author's word counts
    for word in author_word_counts:
        if word in inv_idx and word in idf:  # Check for word in both indexes
            for name, count in inv_idx[word]:
                dot = author_word_counts[word] * count * idf[word]**2
                if name in doc_scores:
                    doc_scores[name] += dot
                else:
                    doc_scores[name] = dot
    return doc_scores


def index_search(
    query_author_word_counts: str,
    index: dict,
    idf,
    doc_norms,
    score_func=accumulate_dot_scores
):
    """Search the collection of documents for the given query

    Arguments
    =========

    query: string,
        The query we are looking for.

    index: an inverted index as above

    idf: idf values precomputed as above

    doc_norms: document norms as computed above

    score_func: function,
        A function that computes the numerator term of cosine similarity (the dot product) for all documents.
        Takes as input a dictionary of query word counts, the inverted index, and precomputed idf values.
        (See Q7)

    tokenizer: a TreebankWordTokenizer

    Returns
    =======

    results: list of tuples (score, doc_id)
        Sorted list of results such that the first element has
        the highest score, and `doc_id` points to the document
        with the highest score.
    """
    # calculate query norm, same formula as doc norm
    query_norm = 0
    for key in query_author_word_counts:
        if key in idf:
            sum_of = query_author_word_counts[key] * idf[key]
            query_norm += sum_of**2
    query_norm = math.sqrt(query_norm)

    # put together to get cossim for each author
    result = []
    num = score_func(query_author_word_counts, index, idf)
    for doc_id in num:
        cossim = num[doc_id] / (doc_norms[doc_id] * query_norm)
        result.append((cossim, doc_id))

    result = sorted(result, key=lambda x: x[0], reverse=True)
    return result

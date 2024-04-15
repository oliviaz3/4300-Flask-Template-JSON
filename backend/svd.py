import json
import os
import numpy as np
from collections import Counter
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
import math
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

def create_docs(data):
  documents = []
  for name in data:
    review_text = ""
    for tok in data[name]["reviews"]:
      review_text += (tok + " ")
    documents.append((name, review_text))
  return documents



def closest_author(documents, author_index_in, author_repr_in, k=100):
  sims = author_repr_in.dot(author_repr_in[author_index_in,:])
  asort = np.argsort(-sims)[:k+1]
  return [(documents[i][0],sims[i]) for i in asort[1:]]


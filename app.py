from flask import Flask, render_template, url_for, flash, redirect, request
import pandas as pd
import numpy as np
from collections import Counter
import math

app = Flask(__name__)

# Load the restaurant data
lko_rest = pd.read_csv("food1.csv")

# Function to calculate term frequency (TF)
def compute_tf(text):
    tf = Counter(text.split())
    tf_sum = sum(tf.values())
    for word in tf:
        tf[word] = tf[word] / tf_sum
    return tf

# Function to calculate inverse document frequency (IDF)
def compute_idf(corpus):
    idf = {}
    total_documents = len(corpus)
    for document in corpus:
        for word in set(document.split()):
            if word not in idf:
                count = sum(1 for doc in corpus if word in doc.split())
                idf[word] = math.log(total_documents / (1 + count))  # Adding 1 to avoid division by zero
    return idf

# Function to calculate TF-IDF
def compute_tfidf(corpus):
    tfidf_matrix = []
    idf = compute_idf(corpus)
    for document in corpus:
        tf = compute_tf(document)
        tfidf = {word: tf[word] * idf.get(word, 0) for word in tf}
        tfidf_matrix.append(tfidf)
    return tfidf_matrix, idf

# Function to compute cosine similarity
def cosine_similarity_manual(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[word] * vec2[word] for word in intersection])
    
    sum1 = sum([vec1[word] ** 2 for word in vec1])
    sum2 = sum([vec2[word] ** 2 for word in vec2])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    
    if denominator == 0:
        return 0.0
    else:
        return numerator / denominator

# Prepare the corpus and calculate TF-IDF matrix for the 'highlights' column
corpus = lko_rest['highlights'].fillna('').tolist()
tfidf_matrix, idf = compute_tfidf(corpus)

# Function for text search based on a query
def search_by_text(query, n_recommendations=10):
    # Compute the TF-IDF for the query
    query_tfidf = compute_tfidf([query])[0][0]
    
    # Compute cosine similarity between the query and each restaurant's highlights
    cosine_scores = []
    for idx, restaurant_tfidf in enumerate(tfidf_matrix):
        score = cosine_similarity_manual(query_tfidf, restaurant_tfidf)
        cosine_scores.append((idx, score))
    
    # Get top restaurant indices
    cosine_scores = sorted(cosine_scores, key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in cosine_scores[:n_recommendations]]
    
    # Get the recommended restaurants
    recommended_restaurants = lko_rest.iloc[top_indices]
    
    # Format results similar to the `calc` function
    rest_list1 = recommended_restaurants.loc[:, 
                ['name', 'address', 'locality', 'timings', 'aggregate_rating', 'url', 'cuisines']]
    rest_list = pd.DataFrame(rest_list1)
    rest_list = rest_list.reset_index()
    rest_list = rest_list.rename(columns={'index': 'res_id'})
    rest_list.drop('res_id', axis=1, inplace=True)
    rest_list = rest_list.T
    ans = rest_list.to_dict()
    res = [value for value in ans.values()]
    return res

def fav(lko_rest1):
    lko_rest1 = lko_rest1.reset_index()
    count1 = CountVectorizer(stop_words='english')
    count_matrix = count1.fit_transform(lko_rest1['highlights'])
    cosine_sim2 = cosine_similarity_manual(count_matrix, count_matrix)

    sim = list(enumerate(cosine_sim2[0]))
    sim = sorted(sim, key=lambda x: x[1], reverse=True)
    sim = sim[1:11]
    indi = [i[0] for i in sim]

    final = lko_rest1.copy().iloc[indi[0]]
    final = pd.DataFrame(final)
    final = final.T

    for i in range(1, len(indi)):
        final1 = lko_rest1.copy().iloc[indi[i]]
        final1 = pd.DataFrame(final1)
        final1 = final1.T
        final = pd.concat([final, final1])

    return final


def rest_rec(cost, people=2, min_cost=0, cuisine=[], Locality=[], fav_rest="", lko_rest=lko_rest):
    cost = cost + 200

    x = cost / people
    y = min_cost / people

    lko_rest1 = lko_rest.copy().loc[lko_rest['locality'] == Locality[0]]

    for i in range(1, len(Locality)):
        lko_rest2 = lko_rest.copy().loc[lko_rest['locality'] == Locality[i]]
        lko_rest1 = pd.concat([lko_rest1, lko_rest2])
        lko_rest1.drop_duplicates(subset='name', keep='last', inplace=True)

    lko_rest_locale = lko_rest1.copy()

    lko_rest_locale = lko_rest_locale.loc[lko_rest_locale['average_cost_for_one'] <= x]
    lko_rest_locale = lko_rest_locale.loc[lko_rest_locale['average_cost_for_one'] >= y]

    lko_rest_locale['Start'] = lko_rest_locale['cuisines'].str.find(cuisine[0])
    lko_rest_cui = lko_rest_locale.copy().loc[lko_rest_locale['Start'] >= 0]

    for i in range(1, len(cuisine)):
        lko_rest_locale['Start'] = lko_rest_locale['cuisines'].str.find(cuisine[i])
        lko_rest_cu = lko_rest_locale.copy().loc[lko_rest_locale['Start'] >= 0]
        lko_rest_cui = pd.concat([lko_rest_cui, lko_rest_cu])
        lko_rest_cui.drop_duplicates(subset='name', keep='last', inplace=True)

    if fav_rest != "":

        favr = lko_rest.loc[lko_rest['name'] == fav_rest].drop_duplicates()
        favr = pd.DataFrame(favr)
        lko_rest3 = pd.concat([favr, lko_rest_cui])
        lko_rest3.drop('Start', axis=1, inplace=True)
        rest_selected = fav(lko_rest3)
    else:
        lko_rest_cui = lko_rest_cui.sort_values('scope', ascending=False)
        rest_selected = lko_rest_cui.head(10)
    return rest_selected


def calc(max_Price, people, min_Price, cuisine, locality):
    rest_sugg = rest_rec(max_Price, people, min_Price, [cuisine], [locality])
    rest_list1 = rest_sugg.copy().loc[:,
                 ['name', 'address', 'locality', 'timings', 'aggregate_rating', 'url', 'cuisines']]
    rest_list = pd.DataFrame(rest_list1)
    rest_list = rest_list.reset_index()
    rest_list = rest_list.rename(columns={'index': 'res_id'})
    rest_list.drop('res_id', axis=1, inplace=True)
    rest_list = rest_list.T
    rest_list = rest_list
    ans = rest_list.to_dict()
    res = [value for value in ans.values()]
    return res


@app.route("/")
@app.route("/home", methods=['GET'])
def home():
    return render_template('home.html')


@app.route("/search", methods=['POST'])
def search():
    if request.method == 'POST':
        people = int(request.form['people'])
        min_Price = int(request.form['min_Price'])
        max_Price =int(request.form['max_Price'])
        cuisine1 = request.form['cuisine']
        locality1 = request.form['locality']
        res = calc(max_Price, people, min_Price,cuisine1, locality1)
        return render_template('search.html', title='Search', restaurants=res)
        #return res
    else:
        return redirect(url_for('home'))

# Add this new route for text search
@app.route("/text_search", methods=['POST'])
def text_search():
    if request.method == 'POST':
        query = request.form['search_text']
        res = search_by_text(query)
        return render_template('search.html', title='Search Results', restaurants=res)
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)

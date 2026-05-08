import pandas as pd
import numpy as np
import re


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse, mae


# ==========================================
# READ DATASET FILES
# ==========================================

movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")


# ==========================================
# DATA PREPROCESSING
# ==========================================

movies.drop_duplicates(inplace=True)
ratings.drop_duplicates(inplace=True)
movies.dropna(inplace=True)
ratings.dropna(inplace=True)


# ==========================================
# MERGE DATASETS
# ==========================================

movie_data = pd.merge(ratings, movies, on='movieId')


# ==========================================
# CONTENT-BASED FILTERING
# ==========================================

movies['genres'] = movies['genres'].fillna('')
movies['content'] = movies['title'] + " " + movies['genres']

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['content'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


# ==========================================
# CREATE SEARCHABLE TITLE MAPPING
# ==========================================

def normalize_title(title):
    """Remove year and convert to lowercase for matching"""
    # Remove year like (1995) from title
    import re
    title_no_year = re.sub(r'\s*\(\d{4}\)', '', title)
    return title_no_year.lower().strip()


# Create multiple indices for flexible matching
movies['title_lower'] = movies['title'].str.lower()
movies['title_no_year'] = movies['title'].apply(normalize_title)

# Index by exact title (case insensitive)
indices_exact = pd.Series(movies.index, index=movies['title_lower']).drop_duplicates()

# Index by title without year
indices_no_year = pd.Series(movies.index, index=movies['title_no_year']).drop_duplicates()


def find_movie(title_input):
    """Find movie index with flexible matching"""
    title_input = title_input.lower().strip()
    
    # Try exact match first
    if title_input in indices_exact.index:
        return indices_exact[title_input]
    
    # Try without year
    title_no_year = re.sub(r'\s*\(\d{4}\)', '', title_input).strip()
    if title_no_year in indices_no_year.index:
        return indices_no_year[title_no_year]
    
    # Try partial match (if input is contained in movie title)
    matches = movies[movies['title_lower'].str.contains(title_input, na=False)]
    if not matches.empty:
        return matches.index[0]
    
    # Try matching on words
    words = title_input.split()
    for word in words:
        if len(word) > 3:
            matches = movies[movies['title_lower'].str.contains(word, na=False)]
            if not matches.empty:
                return matches.index[0]
    
    return None


# ==========================================
# RECOMMENDATION FUNCTIONS
# ==========================================

def content_recommendations_by_movie(title, top_n=10):
    try:
        idx = find_movie(title)
        if idx is None:
            return []
        
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]
        movie_indices = [i[0] for i in sim_scores]
        return movies['title'].iloc[movie_indices].tolist()
    except:
        return []


def content_recommendations_by_genre(genre, top_n=10):
    genre_movies = movies[movies['genres'].str.contains(genre, na=False)]
    avg_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
    avg_ratings.columns = ['movieId', 'avg_rating']
    genre_movies = genre_movies.merge(avg_ratings, on='movieId', how='left')
    genre_movies['avg_rating'] = genre_movies['avg_rating'].fillna(0)
    genre_movies = genre_movies.sort_values('avg_rating', ascending=False)
    return genre_movies.head(top_n)['title'].tolist()


def content_recommendations(search_value, search_type, top_n=10):
    if search_type == "Genre":
        return content_recommendations_by_genre(search_value, top_n)
    else:
        return content_recommendations_by_movie(search_value, top_n)


# ==========================================
# COLLABORATIVE FILTERING (SVD)
# ==========================================

reader = Reader(rating_scale=(0.5, 5))

data = Dataset.load_from_df(
    ratings[['userId', 'movieId', 'rating']],
    reader
)

trainset, testset = train_test_split(
    data,
    test_size=0.2,
    random_state=42
)

svd_model = SVD()

svd_model.fit(trainset)

predictions = svd_model.test(testset)

rmse_score = rmse(predictions)
mae_score = mae(predictions)


def collaborative_recommendations_by_user(user_id, top_n=10):
    rated_movies = ratings[ratings['userId'] == user_id]['movieId'].tolist()
    unrated_movies = movies[~movies['movieId'].isin(rated_movies)]['movieId'].tolist()
    
    predictions_list = []
    for movie_id in unrated_movies:
        pred = svd_model.predict(user_id, movie_id)
        predictions_list.append((movie_id, pred.est))
    
    predictions_list.sort(key=lambda x: x[1], reverse=True)
    top_movies = predictions_list[:top_n]
    
    results = []
    for movie_id, pred_rating in top_movies:
        movie_title = movies[movies['movieId'] == movie_id]['title'].iloc[0]
        results.append(movie_title)
    
    return results


def collaborative_recommendations_by_genre(genre, top_n=10):
    genre_movies = movies[movies['genres'].str.contains(genre, na=False)]
    avg_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
    avg_ratings.columns = ['movieId', 'avg_rating']
    genre_movies = genre_movies.merge(avg_ratings, on='movieId', how='left')
    genre_movies['avg_rating'] = genre_movies['avg_rating'].fillna(0)
    genre_movies = genre_movies.sort_values('avg_rating', ascending=False)
    return genre_movies.head(top_n)['title'].tolist()


def collaborative_recommendations(search_value, search_type, top_n=10):
    if search_type == "Genre":
        return collaborative_recommendations_by_genre(search_value, top_n)
    else:
        try:
            user_id = int(search_value)
            return collaborative_recommendations_by_user(user_id, top_n)
        except:
            return []


# ==========================================
# HYBRID RECOMMENDATION SYSTEM
# ==========================================

def hybrid_recommendations_by_movie(user_id, movie_title, top_n=10):
    try:
        idx = find_movie(movie_title)
        if idx is None:
            return []
        
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:50]
        movie_indices = [i[0] for i in sim_scores]
        
        content_movies = movies.iloc[movie_indices].copy()
        content_movies['content_score'] = [score[1] for score in sim_scores]
        
        def get_collab_score(row):
            pred = svd_model.predict(user_id, row['movieId'])
            return pred.est
        
        content_movies['collaborative_score'] = content_movies.apply(get_collab_score, axis=1)
        content_movies['hybrid_score'] = (
            0.5 * content_movies['content_score'] +
            0.5 * content_movies['collaborative_score']
        )
        
        recommendations = content_movies.sort_values('hybrid_score', ascending=False)
        return recommendations['title'].head(top_n).tolist()
    except Exception as e:
        return []


def hybrid_recommendations_by_genre(genre, top_n=10):
    genre_movies = movies[movies['genres'].str.contains(genre, na=False)]
    avg_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
    avg_ratings.columns = ['movieId', 'avg_rating']
    genre_movies = genre_movies.merge(avg_ratings, on='movieId', how='left')
    genre_movies['avg_rating'] = genre_movies['avg_rating'].fillna(0)
    genre_movies = genre_movies.sort_values('avg_rating', ascending=False)
    return genre_movies.head(top_n)['title'].tolist()


def hybrid_recommendations(user_id, search_value, search_type, top_n=10):
    if search_type == "Genre":
        return hybrid_recommendations_by_genre(search_value, top_n)
    else:
        return hybrid_recommendations_by_movie(user_id, search_value, top_n)


# ==========================================
# GET ALL MOVIES AND GENRES
# ==========================================

def get_all_movies():
    return movies['title'].tolist()


def get_all_genres():
    all_genres = set()
    for genres_str in movies['genres']:
        for genre in genres_str.split('|'):
            all_genres.add(genre)
    return sorted(list(all_genres))


def get_movie_genre(movie_title):
    try:
        movie = movies[movies['title'].str.lower() == movie_title.lower()]
        if not movie.empty:
            return movie.iloc[0]['genres']
        return "Unknown"
    except:
        return "Unknown"


# ==========================================
# GET USER RATINGS COUNT
# ==========================================

def get_user_ratings_count(user_id):
    return len(ratings[ratings['userId'] == user_id])


# ==========================================
# EVALUATION METRICS
# ==========================================

def get_rmse():
    return rmse_score


def get_mae():
    return mae_score


def get_precision_recall_f1(k=10, threshold=4.0):
    user_est_true = {}
    for uid, _, true_r, est, _ in predictions:
        if uid not in user_est_true:
            user_est_true[uid] = []
        user_est_true[uid].append((est, true_r))
    
    precisions = []
    recalls = []
    f1_scores = []
    
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = user_ratings[:k]
        relevant = sum(1 for (_, true_r) in user_ratings if true_r >= threshold)
        recommended_relevant = sum(1 for (_, true_r) in top_k if true_r >= threshold)
        
        prec = recommended_relevant / k if k > 0 else 0
        rec = recommended_relevant / relevant if relevant > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)
    
    return np.mean(precisions), np.mean(recalls), np.mean(f1_scores)
import streamlit as st
import pandas as pd
import plotly.express as px

import HMRS_1 as recommender


st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


st.markdown("""
    <style>
    .stButton > button {
        background-color: #e74c3c;
        color: white;
        border-radius: 20px;
        padding: 10px 25px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #c0392b;
        color: white;
    }
    .stTextInput > div > div > input {
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)


st.title("🎬 Movie Recommendation System")
st.markdown("---")


with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/movie-projector.png", width=80)
    st.title("📊 Dataset Info")
    
    movies_df = pd.read_csv("movies.csv")
    ratings_df = pd.read_csv("ratings.csv")
    
    st.metric("Total Movies", len(movies_df))
    st.metric("Total Users", ratings_df['userId'].nunique())
    st.metric("Total Ratings", f"{len(ratings_df):,}")
    
    st.markdown("---")
    st.markdown("### 🎯 Model Performance")
    
    st.metric("RMSE", f"{recommender.get_rmse():.4f}")
    st.metric("MAE", f"{recommender.get_mae():.4f}")
    
    precision, recall, f1 = recommender.get_precision_recall_f1()
    st.metric("Precision@10", f"{precision:.4f}")
    st.metric("Recall@10", f"{recall:.4f}")
    st.metric("F1-Score@10", f"{f1:.4f}")
    
    st.markdown("---")
    st.markdown("### 📈 Rating Distribution")
    rating_counts = ratings_df['rating'].value_counts().sort_index()
    fig = px.bar(x=rating_counts.index, y=rating_counts.values, 
                 labels={'x': 'Rating', 'y': 'Count'})
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)


tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Content-Based",
    "👥 Collaborative",
    "🔥 Hybrid",
    "📊 Statistics"
])


# ==========================================
# TAB 1: CONTENT-BASED
# ==========================================
with tab1:
    st.header("Find movies similar to your favorite")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_type = st.radio(
            "Search by:",
            ["Movie Name", "Genre"],
            horizontal=True,
            key="content_search"
        )
        
        if search_type == "Movie Name":
            search_value = st.text_input(
                "Enter movie name:",
                placeholder="Example: Toy Story, The Matrix, Inception...",
                key="content_movie"
            )
        else:
            search_value = st.text_input(
                "Enter genre:",
                placeholder="Example: Action, Comedy, Drama, Horror...",
                key="content_genre"
            )
    
    with col2:
        top_n = st.number_input(
            "Number of recommendations:",
            min_value=1,
            value=10,
            step=1,
            key="content_n"
        )
    
    if st.button("🔍 Get Recommendations", key="content_btn"):
        if not search_value:
            st.error("Please enter a movie name or genre!")
        else:
            with st.spinner("Finding recommendations..."):
                results = recommender.content_recommendations(search_value, search_type, top_n)
                
                if results:
                    st.success(f"🎯 Top {len(results)} Recommendations:")
                    for i, movie in enumerate(results, 1):
                        genre = recommender.get_movie_genre(movie)
                        st.markdown(f"""
                        <div style="background-color:#2c3e50; border-radius:10px; padding:15px; margin:10px 0; color:white;">
                            <h3>{i}. 🎬 {movie}</h3>
                            <p><strong>Genres:</strong> {genre}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.error(f"'{search_value}' not found! Please check the spelling.")


# ==========================================
# TAB 2: COLLABORATIVE
# ==========================================
with tab2:
    st.header("Personalized recommendations for you")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_type = st.radio(
            "Search by:",
            ["User ID", "Genre"],
            horizontal=True,
            key="collab_search"
        )
        
        if search_type == "User ID":
            max_user = ratings_df['userId'].max()
            search_value = st.number_input(
                "Enter User ID:",
                min_value=1,
                max_value=max_user,
                value=1,
                step=1,
                key="collab_user"
            )
            user_ratings = recommender.get_user_ratings_count(search_value)
            st.info(f"📊 User {search_value} has rated {user_ratings} movies")
        else:
            search_value = st.text_input(
                "Enter genre:",
                placeholder="Example: Action, Comedy, Drama, Horror...",
                key="collab_genre"
            )
    
    with col2:
        top_n = st.number_input(
            "Number of recommendations:",
            min_value=1,
            value=10,
            step=1,
            key="collab_n"
        )
    
    if st.button("👤 Get Recommendations", key="collab_btn"):
        if not search_value:
            st.error("Please enter a User ID or genre!")
        else:
            with st.spinner("Finding recommendations..."):
                results = recommender.collaborative_recommendations(str(search_value), search_type, top_n)
                
                if results:
                    st.success(f"🎯 Top {len(results)} Recommendations:")
                    for i, movie in enumerate(results, 1):
                        genre = recommender.get_movie_genre(movie)
                        st.markdown(f"""
                        <div style="background-color:#2c3e50; border-radius:10px; padding:15px; margin:10px 0; color:white;">
                            <h3>{i}. 🎬 {movie}</h3>
                            <p><strong>Genres:</strong> {genre}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.error("No recommendations found!")


# ==========================================
# TAB 3: HYBRID
# ==========================================
with tab3:
    st.header("Hybrid recommendations (Content + Collaborative)")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_id = st.number_input(
            "Enter User ID:",
            min_value=1,
            max_value=ratings_df['userId'].max(),
            value=1,
            step=1,
            key="hybrid_user"
        )
        
        search_type = st.radio(
            "Search by:",
            ["Movie Name", "Genre"],
            horizontal=True,
            key="hybrid_search"
        )
        
        if search_type == "Movie Name":
            search_value = st.text_input(
                "Enter movie name:",
                placeholder="Example: Toy Story, The Matrix, Inception...",
                key="hybrid_movie"
            )
        else:
            search_value = st.text_input(
                "Enter genre:",
                placeholder="Example: Action, Comedy, Drama, Horror...",
                key="hybrid_genre"
            )
    
    with col2:
        top_n = st.number_input(
            "Number of recommendations:",
            min_value=1,
            value=10,
            step=1,
            key="hybrid_n"
        )
    
    if st.button("🔥 Get Hybrid Recommendations", key="hybrid_btn"):
        if not search_value:
            st.error("Please enter a movie name or genre!")
        else:
            with st.spinner("Generating hybrid recommendations..."):
                results = recommender.hybrid_recommendations(user_id, search_value, search_type, top_n)
                
                if results:
                    st.success(f"🎯 Top {len(results)} Hybrid Recommendations:")
                    for i, movie in enumerate(results, 1):
                        genre = recommender.get_movie_genre(movie)
                        st.markdown(f"""
                        <div style="background-color:#2c3e50; border-radius:10px; padding:15px; margin:10px 0; color:white;">
                            <h3>{i}. 🎬 {movie}</h3>
                            <p><strong>Genres:</strong> {genre}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.error(f"'{search_value}' not found! Please check the spelling.")


# ==========================================
# TAB 4: STATISTICS
# ==========================================
with tab4:
    st.header("📊 System Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎬 Total Movies", len(movies_df))
    
    with col2:
        st.metric("👥 Total Users", ratings_df['userId'].nunique())
    
    with col3:
        st.metric("⭐ Total Ratings", f"{len(ratings_df):,}")
    
    with col4:
        st.metric("📊 Avg Rating", f"{ratings_df['rating'].mean():.2f}")
    
    st.markdown("---")
    
    st.subheader("🎭 Top Genres")
    
    genre_counts = {}
    for genres_str in movies_df['genres']:
        for genre in genres_str.split('|'):
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
    
    genre_df = pd.DataFrame(list(genre_counts.items()), columns=['Genre', 'Count'])
    genre_df = genre_df.sort_values('Count', ascending=False).head(15)
    
    fig = px.bar(genre_df, x='Genre', y='Count', title='Top 15 Genres', color='Count')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("⭐ Top Rated Movies")
    
    avg_ratings = ratings_df.groupby('movieId')['rating'].mean().reset_index()
    avg_ratings = avg_ratings.merge(movies_df[['movieId', 'title', 'genres']], on='movieId')
    avg_ratings = avg_ratings.sort_values('rating', ascending=False).head(20)
    
    fig = px.bar(avg_ratings, x='rating', y='title', orientation='h',
                 title='Top 20 Movies by Average Rating',
                 color='rating',
                 labels={'rating': 'Average Rating', 'title': 'Movie Title'})
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)


st.markdown("---")
st.markdown("### 📊 Model Performance Summary")
st.markdown(f"""
| Metric | Value |
|--------|-------|
| RMSE | {recommender.get_rmse():.4f} |
| MAE | {recommender.get_mae():.4f} |
| Precision@10 | {precision:.4f} |
| Recall@10 | {recall:.4f} |
| F1-Score@10 | {f1:.4f} |
""")
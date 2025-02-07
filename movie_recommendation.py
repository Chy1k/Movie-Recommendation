import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load the ratings data
data = pd.read_csv('./ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
# Load movie titles
movies = pd.read_csv('./ml-100k/u.item', sep='|', names=['movie_id', 'title'], usecols=[0, 1], encoding='latin-1')
# Drop the 'timestamp' column as we don't need it
data = data.drop('timestamp', axis=1)

# Create the user-item matrix with users as rows and movies as columns
user_item_matrix = data.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
# Create a dictionary to map movie IDs to titles
movie_dict = pd.Series(movies.title.values, index=movies.movie_id).to_dict()

# Calculate cosine similarity between users
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# Get the user's similarity scores with other users
def recommend_movies(user_id, num_recommendations=10):
    # Ensure the user exists in the similarity matrix
    if user_id not in user_similarity_df.index:
        print(f"User ID {user_id} not found.")
        return None

    # Step 1: Sort similar users
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)
    print(f"Similar users for User {user_id}:\n{similar_users.head()}")

    # Step 2: Gather recommendations
    recommendations = []
    for similar_user in similar_users.index[1:]: # Skip the first since it's the user themselves
        user_ratings = user_item_matrix.loc[similar_user]

        # Lower threshold to 3 if no recommendations show up with 4
        user_recommendations = user_ratings[user_ratings > 4]
        print(f"Movies rated 3 or above by User {similar_user}:\n{user_recommendations}")

        recommendations.extend(user_recommendations.index.tolist())

        # Stop once we have enough recommendations
        if len(recommendations) >= num_recommendations:
            break

    # Step 3: Get unique recommendations and map to titles
    unique_recommendations = set(recommendations[:num_recommendations])
    recommended_titles = [movie_dict.get(movie_id, "Unknown title") for movie_id in unique_recommendations]
    print(f"Recommended Movie IDs for User {user_id}: {unique_recommendations}")
    print(f"Recommended Movie Titles for User {user_id}: {recommended_titles}")


    # Return unique recommendations
    return recommended_titles if recommended_titles else None

# Call the function and apply the display Style
recommendations = recommend_movies(1) # Get recommendations for User 1

# Emoji Style
if recommendations:
    print("📽️ Recommended Movies for User 1 📽️")
    for idx, title in enumerate(recommendations, start=1):
        print(f"🌟 {idx}.  {title}")
else:
    print("No recommendations found.")


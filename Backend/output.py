import pandas as pd
import streamlit as st

@st.cache_data 
def get_user_info():
    # Find the top 10 AuthorId with the most reviews
    top_10_authors = reviews['AuthorId'].value_counts().nlargest(10)
    top_10_author_ids = top_10_authors.index.tolist()

    # Create a DataFrame to store the results
    user_info = pd.DataFrame(columns=['AuthorId', 'AuthorName', 'TotalRatings', 'Rating1', 'Rating2', 'Rating3', 'Rating4', 'Rating5'])

    for author_id in top_10_author_ids:
        author_name = reviews.loc[reviews['AuthorId'] == author_id, 'AuthorName'].values[0]
        total_ratings = len(reviews[reviews['AuthorId'] == author_id])

        # Count the occurrences of each rating for the author
        rating_counts = reviews[reviews['AuthorId'] == author_id]['Rating'].value_counts().sort_index()

        # Create a row for the result DataFrame
        result_row = {
            'AuthorId': author_id,
            'AuthorName': author_name,
            'TotalRatings': total_ratings,
            'Rating1': rating_counts.get(1, 0),
            'Rating2': rating_counts.get(2, 0),
            'Rating3': rating_counts.get(3, 0),
            'Rating4': rating_counts.get(4, 0),
            'Rating5': rating_counts.get(5, 0)
        }

        # Append the row to the result DataFrame
        user_info = pd.concat([user_info, pd.DataFrame([result_row])], ignore_index=True)
    return user_info

@st.cache_data 
def get_users():
    return user_info['AuthorId'].tolist()

@st.cache_data 
def get_name(selected_author_id):
    selected_author_row = user_info[user_info['AuthorId'] == selected_author_id]
    if not selected_author_row.empty:
        return selected_author_row['AuthorName'].values[0]
    else:
        return f"Author {selected_author_id} not found"

@st.cache_data 
def get_ratings(selected_author_id):
    selected_author_row = user_info[user_info['AuthorId'] == selected_author_id]
    if not selected_author_row.empty:
        return selected_author_row['TotalRatings'].values[0]
    else:
        return f"Author {selected_author_id} not found"

@st.cache_data 
def get_breakdown(selected_author_id):
    selected_author_row = user_info[user_info['AuthorId'] == selected_author_id]
    if not selected_author_row.empty:
        return [
            int(selected_author_row['Rating1'].values[0]),
            int(selected_author_row['Rating2'].values[0]),
            int(selected_author_row['Rating3'].values[0]),
            int(selected_author_row['Rating4'].values[0]),
            int(selected_author_row['Rating5'].values[0])
        ]
    else:
        return [0, 0, 0, 0, 0]

@st.cache_data 
def get_reviews(selected_author_id):
    selected_author_reviews = reviews[reviews['AuthorId'] == selected_author_id][['RecipeId', 'Name', 'Rating', 'Review', 'DateSubmitted', 'DateModified']].copy()
    selected_author_reviews = selected_author_reviews.sort_values(by='DateModified', ascending=False).head(5)
    return selected_author_reviews

reviews = pd.read_csv('Data/reviews_cleaned.csv')
recipes = pd.read_csv('Data/recipes_cleaned.csv')

# Find the top 10 AuthorId with the most reviews
top_10_authors = reviews['AuthorId'].value_counts().nlargest(10)
top_10_author_ids = top_10_authors.index.tolist()

# Create a DataFrame to store the results
user_info = pd.DataFrame(columns=['AuthorId', 'AuthorName', 'TotalRatings', 'Rating1', 'Rating2', 'Rating3', 'Rating4', 'Rating5'])

for author_id in top_10_author_ids:
    author_name = reviews.loc[reviews['AuthorId'] == author_id, 'AuthorName'].values[0]
    total_ratings = len(reviews[reviews['AuthorId'] == author_id])

    # Count the occurrences of each rating for the author
    rating_counts = reviews[reviews['AuthorId'] == author_id]['Rating'].value_counts().sort_index()

    # Create a row for the result DataFrame
    result_row = {
        'AuthorId': author_id,
        'AuthorName': author_name,
        'TotalRatings': total_ratings,
        'Rating1': rating_counts.get(1, 0),
        'Rating2': rating_counts.get(2, 0),
        'Rating3': rating_counts.get(3, 0),
        'Rating4': rating_counts.get(4, 0),
        'Rating5': rating_counts.get(5, 0)
    }

    # Append the row to the result DataFrame
    user_info = pd.concat([user_info, pd.DataFrame([result_row])], ignore_index=True)
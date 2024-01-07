import nltk
import pickle
import pandas as pd
import numpy as np
import streamlit as st
from surprise import Dataset, Reader, SVD
from sklearn.metrics.pairwise import linear_kernel
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

# Model 1 - SVD User-Based Collaborate Filtering
@st.cache_data
def model1(selected_author_id, top_n=20):
    # Load the SVD model from the pickle file
    svd_model_filename = 'Resources/svd_model.pkl'
    with open(svd_model_filename, 'rb') as svd_file:
        svd_model_pkl = pickle.load(svd_file)

    rated_reviews = reviews_train[reviews_train['AuthorId'] == selected_author_id]
    unrated_reviews = reviews_train[reviews_train['AuthorId'] != selected_author_id]

    # Get the list of all unique items (RecipeId) that the user has rated
    rated_items = rated_reviews['RecipeId'].unique()

    # Get the items that the user hasn't rated
    unrated_items = unrated_reviews['RecipeId'].unique()

    # Predict ratings for the unrated items
    svd_user_ratings = [(selected_author_id, RecipeId, svd_model_pkl.predict(selected_author_id, RecipeId).est) for RecipeId in unrated_items]

    # Sort the predicted ratings in descending order
    svd_user_ratings.sort(key=lambda x: x[2], reverse=True)

    # Get the top N recommendations
    svd_result = svd_user_ratings[:top_n]

    # Create svd_result_df DataFrame
    svd_result_df = pd.DataFrame({
        'RecipeId': [RecipeId for _, RecipeId, _ in svd_result],
        'Name': [recipes.loc[recipes['RecipeId'] == int(RecipeId)].iloc[0]['Name'] for _, RecipeId, _ in svd_result],
    })
    return svd_result_df

# Model 2 - Word2Vec Content-Based Filtering
@st.cache_data 
def model2(selected_author_id, top_n=20):
    # Create a user profile
    all_recipe_ids = recipes['RecipeId'].unique()
    author_ratings = reviews[(reviews['AuthorId'] == selected_author_id) & (reviews['RecipeId'].isin(all_recipe_ids))]
    average_ratings = author_ratings.groupby('RecipeId')['Rating'].mean().reset_index()
    user_profile_table = pd.merge(recipes, average_ratings, on='RecipeId', how='left')[['RecipeId', 'Rating']]

    # Load the Word2Vec model from the pickle file
    word2vec_model_filename = 'Resources/word2vec_model.pkl'
    with open(word2vec_model_filename, 'rb') as word2vec_file:
        word2vec_model_pkl = pickle.load(word2vec_file)

    # Reset index to avoid KeyError
    sampled_reviews = reviews.sample(frac=0.01, random_state=42)
    sampled_reviews = sampled_reviews.reset_index(drop=True)

    # Drop unnecessary columns
    content_data = sampled_reviews[['AuthorId', 'RecipeId', 'Description', 'Review', 'Name', 'Rating']]

    # Drop rows with missing values in the 'Name' columns
    content_data = content_data.dropna(subset=['Name'])

    # Tokenize the combined text into words
    content_data['combined_text'] = content_data['Review'].fillna('') + ' ' + content_data['Description'].fillna('') + ' ' + content_data['Rating'].astype(str)
    content_data['tokenized_text'] = content_data['combined_text'].apply(word_tokenize)

    # Function to average word vectors for a document
    def average_word_vectors(words, model, vocabulary, num_features):
        feature_vector = np.zeros((num_features,), dtype="float32")
        nwords = 0.
        for word in words:
            if word in vocabulary:
                nwords = nwords + 1.
                feature_vector = np.add(feature_vector, model.wv[word])
        if nwords:
            feature_vector = np.divide(feature_vector, nwords)

        return feature_vector

    # Create a feature matrix by averaging word vectors for each document
    vocabulary = set(word2vec_model_pkl.wv.index_to_key)
    word2vec_num_features = 100  # Change this to match the vector_size parameter in Word2Vec

    content_data['word2vec_features'] = content_data['tokenized_text'].apply(
        lambda x: average_word_vectors(x, word2vec_model_pkl, vocabulary, word2vec_num_features)
    )

    # Create a DataFrame with the feature vectors
    word2vec_features_df = pd.DataFrame(content_data['word2vec_features'].tolist(), index=content_data.index)

    # Concatenate the word2vec_features_df with other relevant columns
    content_data = pd.concat([content_data, word2vec_features_df], axis=1)

    # Calculate cosine similarity between word2vec features
    cosine_sim_word2vec = linear_kernel(word2vec_features_df, word2vec_features_df)

    def get_top_similar_recipes_word2vec(author_id, user_profile=user_profile_table, cosine_sim=cosine_sim_word2vec,
                                        content_data=content_data, N=top_n):
        # Get the index of the selected author in the content_data
        idx = content_data[content_data['AuthorId'] == author_id].index

        if not idx.empty:
            idx = idx[0]

            # Calculate cosine similarity between Word2Vec features
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

            # Filter out recipes already reviewed by the selected author
            author_reviews = content_data[content_data['AuthorId'] == author_id]['RecipeId'].tolist()
            sim_scores = [(i, score) for i, score in sim_scores if content_data.iloc[i]['RecipeId'] not in author_reviews]

            # Weigh similarity scores based on user's ratings
            sim_scores = [
                (i, score * (1 + content_data.iloc[i]['Rating'] - user_profile[user_profile['RecipeId'] ==
                                                                        content_data.iloc[i]['RecipeId']]['Rating'].values[0]))
                for i, score in sim_scores
            ]

            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:N]

            # Create a DataFrame with the recommendations
            recommended_recipes = pd.DataFrame(columns=content_data.columns)
            seen_recipe_ids = set()

            for i, _ in sim_scores:
                recipe_id = content_data.iloc[i]['RecipeId']

                # Check if the recipe is not already in the recommended_recipes list
                if recipe_id not in seen_recipe_ids:
                    recommended_recipes = pd.concat([recommended_recipes, content_data.iloc[[i]]])
                    seen_recipe_ids.add(recipe_id)

                # Break if N recommendations are reached
                if len(recommended_recipes) >= N:
                    break

            return recommended_recipes.head(N)
        else:
            print(f"Author with ID {author_id} not found.")
            return pd.DataFrame()

    # Use the updated function to get top similar recipes with Word2Vec
    word2vec_result = get_top_similar_recipes_word2vec(selected_author_id)
    word2vec_result_df = word2vec_result[["RecipeId", "Name"]].reset_index(drop=True)
    return word2vec_result_df

# Final Hybrid Model
@st.cache_data 
def final_model(selected_author_id, svd_result_df, word2vec_result_df, final_n=20):
    # Load the SVD model from the pickle file
    svd_model_filename = 'Resources/svd_model.pkl'
    with open(svd_model_filename, 'rb') as svd_file:
        svd_model_pkl = pickle.load(svd_file)

    
    # combined_recommendations = [RecipeId for _, RecipeId, _ in svd_result] + word2vec_result['RecipeId'].tolist()
    combined_recommendations = svd_result_df['RecipeId'].tolist() + word2vec_result_df['RecipeId'].tolist()

    # Remove duplicates, if any
    combined_recommendations = list(set(combined_recommendations))

    # Sort the combined list based on predicted ratings from SVD model
    combined_recommendations.sort(key=lambda x: svd_model_pkl.predict(selected_author_id, x).est, reverse=True)

    # Select the top 10 recommendations
    hybrid_result = combined_recommendations[:final_n]
    hybrid_result_data = []

    # Populate the data for the hybrid result
    for recipe_id in hybrid_result:
        # Get the predicted score from the SVD model
        predicted_score = svd_model_pkl.predict(selected_author_id, recipe_id).est

        # Get the recipe name
        matched_recipe = recipes.loc[recipes['RecipeId'] == int(recipe_id)]

        # Append data to the list for DataFrame creation
        if not matched_recipe.empty:
            recommended_recipe_name = matched_recipe.iloc[0]['Name']
            hybrid_result_data.append({
                'RecipeId': recipe_id,
                'Name': recommended_recipe_name,
                'PredictedRating': predicted_score
            })

    hybrid_result_df = pd.DataFrame(hybrid_result_data)
    return hybrid_result_df

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
def get_reviews(selected_author_id, n):
    selected_author_reviews = reviews[reviews['AuthorId'] == selected_author_id][['RecipeId', 'Name', 'Rating', 'Review', 'DateSubmitted', 'DateModified']].copy()
    selected_author_reviews = selected_author_reviews.sort_values(by='DateModified', ascending=False).head(n)
    return selected_author_reviews

@st.cache_data 
def get_recipes(result):
    return recipes[recipes['RecipeId'].isin(result['RecipeId'].tolist())]

reviews = pd.read_csv('Data/reviews_cleaned.csv')
recipes = pd.read_csv('Data/recipes_cleaned.csv')

# Split and sample the datasets
reviews_train, reviews_test = train_test_split(reviews, test_size=0.3, random_state=42)

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

# Generate top 10 recommendations for a specific user
# selected_author_id = 169430
# top_n = 10
# print(f"Target User ID: {selected_author_id}")
# print(f"Number of Recommendations: {top_n}")

# svd_result = model1(selected_author_id, top_n)
# print(svd_result)

# word2vec_result = model2(selected_author_id, top_n)
# print(word2vec_result)

# hybrid_result = final_model(selected_author_id, svd_result, word2vec_result, 20)
# print(hybrid_result)

# for meal_name,column,recommendation in zip(meals,st.columns(len(meals)),recommendations):
#     with column:
#         #st.markdown(f'<div style="text-align: center;">{meal_name.upper()}</div>', unsafe_allow_html=True) 
#         st.markdown(f'##### {meal_name.upper()}')    
#         for recipe in recommendation:
            
#             recipe_name=recipe['Name']
#             expander = st.expander(recipe_name)
#             recipe_link=recipe['image_link']
#             recipe_img=f'<div><center><img src={recipe_link} alt={recipe_name}></center></div>'     
#             nutritions_df=pd.DataFrame({value:[recipe[value]] for value in nutritions_values})      
            
#             expander.markdown(recipe_img,unsafe_allow_html=True)  
#             expander.markdown(f'<h5 style="text-align: center;font-family:sans-serif;">Nutritional Values (g):</h5>', unsafe_allow_html=True)                   
#             expander.dataframe(nutritions_df)
#             expander.markdown(f'<h5 style="text-align: center;font-family:sans-serif;">Ingredients:</h5>', unsafe_allow_html=True)
#             for ingredient in recipe['RecipeIngredientParts']:
#                 expander.markdown(f"""
#                             - {ingredient}
#                 """)
#             expander.markdown(f'<h5 style="text-align: center;font-family:sans-serif;">Recipe Instructions:</h5>', unsafe_allow_html=True)    
#             for instruction in recipe['RecipeInstructions']:
#                 expander.markdown(f"""
#                             - {instruction}
#                 """) 
#             expander.markdown(f'<h5 style="text-align: center;font-family:sans-serif;">Cooking and Preparation Time:</h5>', unsafe_allow_html=True)   
#             expander.markdown(f"""
#                     - Cook Time       : {recipe['CookTime']}min
#                     - Preparation Time: {recipe['PrepTime']}min
#                     - Total Time      : {recipe['TotalTime']}min
#                 """)

# # Filter the recipes table based on RecipeId in svd_result
# selected_svd = recipes[recipes['RecipeId'].isin(svd_result['RecipeId'].tolist())]

# # Display the selected recipes
# print(selected_svd)

# selected_svd = recipes[recipes['RecipeId'].isin(svd_result['RecipeId'].tolist())]
# meals = selected_svd["Name"]  
# recommendations = [selected_svd.to_dict(orient="records")] * len(meals) 
# nutritions_values = ["Calories", "Protein", "Carbohydrates", "Fat"] 

# # Loop through meals, recommendations, and nutritions_values
# for meal_name, column, recommendation in zip(meals, st.columns(len(meals)), recommendations):
#     with column:
#         st.markdown(f'##### {meal_name.upper()}')
#         for recipe in recommendation:
#             recipe_name = recipe['Name']
#             expander = st.expander(recipe_name)
            # recipe_link = recipe.get('image_link', '')  # Replace with your actual image link
            # recipe_img = f'<div><center><img src={recipe_link} alt={recipe_name}></center></div>'
            # nutritions_df = pd.DataFrame({value: [recipe.get(value, '')] for value in nutritions_values})

            # expander.markdown(recipe_img, unsafe_allow_html=True)
            # expander.markdown(f'<h5 style="text-align: center;font-family:sans-serif;">Nutritional Values (g):</h5>', unsafe_allow_html=True)
            # expander.dataframe(nutritions_df)
            # expander.markdown(f'<h5 style="text-align: center;font-family:sans-serif;">Ingredients:</h5>', unsafe_allow_html=True)
            # for ingredient in recipe.get('RecipeIngredientParts', []):
            #     expander.markdown(f"- {ingredient}")
            # expander.markdown(f'<h5 style="text-align: center;font-family:sans-serif;">Recipe Instructions:</h5>', unsafe_allow_html=True)
            # for instruction in recipe.get('RecipeInstructions', []):
            #     expander.markdown(f"- {instruction}")
            # expander.markdown(f'<h5 style="text-align: center;font-family:sans-serif;">Cooking and Preparation Time:</h5>', unsafe_allow_html=True)
            # expander.markdown(f"""
            #         - Cook Time       : {recipe.get('CookTime', '')}min
            #         - Preparation Time: {recipe.get('PrepTime', '')}min
            #         - Total Time      : {recipe.get('TotalTime', '')}min
            #     """)
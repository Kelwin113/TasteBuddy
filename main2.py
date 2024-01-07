# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import re

# Main App
st.set_page_config(page_title="TasteBuddy", page_icon="Resources/Icon.ico")
st.sidebar.image("Resources/Logo3.png", use_column_width=True)
page_options = ["Home Page","User Profile","Algorithm","Hybrid Food Recommender"]
page_selection = st.sidebar.selectbox("Choose Option", page_options)

# Custom Libraries
from Backend.model import model1, model2, final_model, get_users, get_name, get_ratings, get_reviews, get_breakdown, get_recipes

# Function to get or create the session state
def get_session_state():
    session_state = st.session_state
    if not hasattr(session_state, 'selected_author_id'):
        session_state.selected_author_id = 169430
    if not hasattr(session_state, 'page'):
        session_state.page = 'home'
    return session_state
    
# App declaration
def home_page():
    # Add content for the home page
    st.markdown("")
    session_state = get_session_state()
    st.title('TasteBuddy: Your Personalized Food Recommender')
    st.write("Welcome to TasteBuddy! Your go-to platform for personalized food recommendations. "
            "Discover a world of unique and delightful flavors tailored just for you.")
    st.image("Resources/Logo2.png", caption='TasteBuddy Logo', use_column_width=True)
    selected_author_id = st.selectbox('Selected User: ', get_users(), index=get_users().index(session_state.selected_author_id))
    st.write(f'Name: {get_name(selected_author_id)}')

    if st.button('Explore Now'):
        st.write('Your Exclusive and Extraordinary Food Adventure Now Begins!')
    
    # Update the session state with the selected author ID
    session_state.selected_author_id = selected_author_id
    
def user_profile():
    # Add content for the user profile page
    session_state = get_session_state()

    st.title(f"Welcome, {get_name(session_state.selected_author_id)} 👋")
    st.write('### Profile Details:')
    st.write(f'User ID: {session_state.selected_author_id}')
    st.write(f'Name: {get_name(session_state.selected_author_id)}')
    st.write(f'Total Reviews/Ratings: {get_ratings(session_state.selected_author_id)}')

    # Plotting the bar chart
    ratings = get_breakdown(session_state.selected_author_id)
    labels = ['Rating 1', 'Rating 2', 'Rating 3', 'Rating 4', 'Rating 5']
    bar_chart_data = dict(zip(labels, ratings))
    st.bar_chart(bar_chart_data)

    st.write('### Past Reviews:')
    reviews_n = st.selectbox("Retrive number of past reviews:", [5,10,15])
    st.table(get_reviews(session_state.selected_author_id, reviews_n))

def hybrid_food_recommender():
    # Add content for the hybrid food recommender page
    session_state = get_session_state()

    st.title("Hybrid Food Recommender")
    st.subheader('Step into Your Exclusive and Extraordinary Food Adventure!')
    final_n = st.slider("Top N Recommendations", min_value=5, max_value=20, value=10)

def algorithm():
    # Add content for the algorithm page
    session_state = get_session_state()
    st.title('Algorithm')

    model_n = st.slider("Top N Recommendations", min_value=5, max_value=20, value=10)

    # Recommender System algorithm selection
    model = st.radio("Select an algorithm",
                    ('Content Based Filtering',
                    'Collaborative Filtering'))

    # Perform recommendation generation
    if model == 'Content Based Filtering':
        if st.button("Recommend"):
            try:
                # with st.spinner('Starting Your Personalized Food Recommendations...'):
                word2vec_result = model2(session_state.selected_author_id)
                st.success("Recommendations generated successfully!")
                st.write(f"Top {model_n} Recommended Food Items for User {session_state.selected_author_id} Using Word2vec Content-Based Filtering:")
                st.table(word2vec_result.head(model_n))
            except Exception as e:
                st.error("Oops! Looks like this algorithm does't work. Please try again later!")
                st.error(f"An error occurred: {str(e)}")

    if model == 'Collaborative Filtering':
        if st.button("Recommend"):
            try:
                # with st.spinner('Starting Your Personalized Food Recommendations...'):
                svd_result = model1(session_state.selected_author_id)
                st.write(f"Top {model_n} Recommended Food Items for User {session_state.selected_author_id} Using SVD Collaborative Filtering:")
                svd_result = svd_result.head(model_n)
                st.table(svd_result)
                svd_result_full = get_recipes(svd_result)

                # Display detailed information for each recommended item
                for index, row in svd_result.iterrows():
                    recipe_id = row['RecipeId']
                    details = svd_result_full[svd_result_full['RecipeId'] == recipe_id]
                    expander = st.expander(f"{row['RecipeName']} (Recipe ID: {recipe_id})")

                    # Display details in the expander
                    # Display image using the provided link
                    image_link = details['Images'].iloc[0]
                    if image_link:
                        expander.image(image_link, caption=row['RecipeName'], use_column_width=True)
                        expander.write(f"Recipe Category: {details['RecipeCategory'].iloc[0]}")

                    # Display nutritional information in a table
                    expander.write("### Nutritional Information:")
                    nutritional_info = details[['Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent',
                                                'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent',
                                                'ProteinContent']]
                    expander.table(nutritional_info)
                    expander.write(f"Description: {details['Description'].iloc[0]}")

                    expander.write("### Recipe Information:")
                    duration_info = details[['CookTime','PrepTime','TotalTime']]
                    expander.table(duration_info)

                    # from PIL import Image
                    # import requests
                    # from io import BytesIO
                    # # Assuming image_url contains the URL of the image
                    # image_url = "https://img.sndimg.com/food/image/upload/w_555,h_416,c_fit,fl_progressive,q_95/v1/img/recipes/12/30/3/picYyZVyR.jpg"

                    # # Fetch the image
                    # response = requests.get(image_url)
                    # image = Image.open(BytesIO(response.content))

                    # # Display the image in the expander
                    # expander.image(image, caption=row['RecipeName'], use_column_width=True)

            except Exception as e:
                st.error("Oops! Looks like this algorithm does't work. Please try again later!")
                st.error(f"An error occurred: {str(e)}")

if page_selection == "Home Page":
    home_page()
if page_selection == "User Profile":
    user_profile()
if page_selection == "Algorithm":
    algorithm()
if page_selection == "Hybrid Food Recommender":
    hybrid_food_recommender()
    
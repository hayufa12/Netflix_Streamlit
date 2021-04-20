import pandas as pd
import plotly.express as px
import streamlit as st
import application_function as func
st.write("""
# Netflix EDA and Movies Similarity

This app explore the **Netflix Dataset** and Find Similiar Movies
""")
st.write('---')
st.header('Description')
st.write("""
This dataset consists of tv shows and movies available on Netflix as of 2019. The dataset is collected from Flixable which is a third-party Netflix search engine.

In 2018, they released an interesting report which shows that the number of TV shows on Netflix has nearly tripled since 2010. The streaming service’s number of movies has decreased by more than 2,000 titles since 2010, while its number of TV shows has nearly tripled. It will be interesting to explore what all other insights can be obtained from the same dataset.

Integrating this dataset with other external datasets such as IMDB ratings, rotten tomatoes can also provide many interesting findings.
""")
if st.button("Show Data"):
    st.write(func.df)
st.header('Exploratory Data Analysis')
column = st.selectbox(
    "Select the column that you want to explore",
    func.df.columns.drop(['rating','type','show_id', 'title','date_added','release_year','duration','description'])
    )
num = st.slider("Select the number of item", 1, 20, 10)
st.write(func.to_bar(column, num))
st.write('---')

st.header('Recommended Movies')
movies = st.selectbox(
    "Select your favorite movie",
    func.movie.title
    )
st.subheader('Movies that are similiar to your favorite one:')
st.write(func.movies_similarity(movies))
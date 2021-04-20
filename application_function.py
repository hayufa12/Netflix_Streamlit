import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('netflix_titles.csv', header=0)
df['cast'] = df['cast'].fillna('NA').str.split(', ')
df['director'] = df['director'].fillna('NA').str.split(', ')
df['country'] = df['country'].fillna('Worldwide').str.split(', ')
df['listed_in'] = df['listed_in'].fillna('NA').str.split(', ')

movie = pd.read_csv('netflix_titles.csv', header=0)
movie['index']=movie.index
features = ['type','director','cast','rating','listed_in','description']
movie['country'].fillna('Worldwide', inplace=True)
movie['cast'] = movie['cast'].str.split(', ')
movie['director'] = movie['director'].str.split(', ')
movie['country'] = movie['country'].str.split(', ')
movie['listed_in'] = movie['listed_in'].str.split(', ')
movie['cast'].dropna(inplace=True)
movie['director'].dropna(inplace=True)

# Movies similarity function
for feature in features:
    movie[feature] = movie[feature].fillna('')

def list_to_string(list):
    return ','.join(map(str, list)) 

def title_from_index(index):
    return movie[movie['index'] == index]["title"].values[0]

def index_from_title(title):
    title_list = movie['title'].tolist()
    common = difflib.get_close_matches(title, title_list, 1)
    titlesim = common[0]
    return movie[movie['title'] == titlesim]["index"].values[0]

for count, row in enumerate(movie['cast']):
    changed = list_to_string(row)
    movie['cast'].iloc[count]=changed

for count, row in enumerate(movie['director']):
    changed = list_to_string(row)
    movie['director'].iloc[count]=changed
    
for count, row in enumerate(movie['listed_in']):
    changed = list_to_string(row)
    movie['listed_in'].iloc[count]=changed

def combine_features(row):
    try:
        return row['type'] +" "+row['director']+" "+row['cast']+" "+row['rating']+" "+row['listed_in']+" "+row['description']
    except:
        print ("Error:", row)

movie["combined_features"] = movie.apply(combine_features,axis=1)

# Model Training
cv = CountVectorizer()
count_matrix = cv.fit_transform(movie["combined_features"])
cosine_sim = cosine_similarity(count_matrix) 

# Function to convert list in a column to 1d list
def to_1D(series):
 return pd.Series([x for _list in series for x in _list])

# Function to display bar chart
def to_bar(column,num):
    fig = px.bar(
        df,
        x=to_1D(df[column]).value_counts(sort=True).iloc[1:num+1].index, 
        y=to_1D(df[column]).value_counts(sort=True).iloc[1:num+1].values, 
        labels={
            'x' : column.capitalize(),
            'y' : 'Number of Movies / Tv Shows'
            },
        title = f'Most Productive {column.capitalize()}')
    return fig

def movies_similarity(user_movie):
    movie_index = index_from_title(user_movie)
    similar_movies =  list(enumerate(cosine_sim[movie_index]))
    final_movies = []
    similar_movies_sorted = sorted(similar_movies,key=lambda x:x[1],reverse=True) 
    i=0 
    for rec_movie in similar_movies_sorted:
        if(i!=0):
            final_movies.append(title_from_index(rec_movie[0]))
        i=i+1
        if i>50:
            break
    final_df = movie[movie['title'].isin(final_movies)][['title','type','director','cast','rating','listed_in','description']].reset_index(drop=True)
    return final_df
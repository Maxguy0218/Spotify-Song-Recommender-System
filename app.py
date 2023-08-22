#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import difflib
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance

# Load data and preprocess it
data = pd.read_csv('genres_v2.csv')
df = data[data.columns[:11]]
df['genre'] = data['genre']
df['time_signature'] = data['time_signature']
df['duration_ms'] = data['duration_ms']
df['song_name'] = data['song_name']
x = df[df.drop(columns=['song_name', 'genre']).columns].values
scaler = StandardScaler().fit(x)
X_scaled = scaler.transform(x)
df[df.drop(columns=['song_name', 'genre']).columns] = X_scaled

def find_word(word, words_list):
    return difflib.get_close_matches(word, words_list, 1)[0]

def make_matrix_cosine(data, song, number):
    df = pd.DataFrame()
    data.drop_duplicates(inplace=True)
    songs = data['song_name'].values
    best = find_word(song, songs)
    st.write('The song closest to your search is :', best)
    genre = data[data['song_name'] == best]['genre'].values[0]
    df = data[data['genre'] == genre]
    x = df[df['song_name'] == best].drop(columns=['genre', 'song_name']).values
    if len(x) > 0:
        x = x[0]  # Extract the first row as a 1-dimensional array
    else:
        st.write("No matching song data found.")
        return
    song_names = df['song_name'].values
    df.drop(columns=['genre', 'song_name'], inplace=True)
    df = df.fillna(df.mean())
    p = []
    count = 0
    for i in df.values:
        p.append([distance.cosine(x, i), count])
        count += 1
    p.sort()
    recommendations = []
    for i in range(1, number + 1):
        recommendations.append(song_names[p[i][1]])
    return recommendations

def main():
    st.title("Spotify Song Recommendation System")
    st.write("Enter the name of a song and the number of recommendations you want.")

    c = st.text_input('Please enter the name of the song:')
    d = st.number_input('Please enter the number of recommendations you want:', min_value=1, step=1)

    if st.button('Recommend'):
        recommendations = make_matrix_cosine(df, c, d)
        st.write("Recommended Songs:")
        for i, recommendation in enumerate(recommendations):
            st.write(f"{i+1}. {recommendation}")

if __name__ == "__main__":
    main()


# In[ ]:





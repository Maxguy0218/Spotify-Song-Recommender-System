#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
import streamlit as st
import copy
import warnings
warnings.filterwarnings("ignore")


# In[2]:


data=pd.read_csv('genres_v2.csv')


# In[3]:


data.head()


# In[4]:


data=data.dropna(subset=['song_name'])


# In[5]:


# Creating a new dataframe with required features
df=data[data.columns[:11]]
df['genre']=data['genre']
df['time_signature']=data['time_signature']
df['duration_ms']=data['duration_ms']
df['song_name']=data['song_name']


# In[6]:


x=df[df.drop(columns=['song_name','genre']).columns].values
scaler = StandardScaler().fit(x)
X_scaled = scaler.transform(x)
df[df.drop(columns=['song_name','genre']).columns]=X_scaled


# In[9]:


# This is a function to find the closest song name from the list
def find_word(word,words):
    t=[]
    count=0
    if word[-1]==' ':
        word=word[:-1]
    for i in words:
        if word.lower() in i.lower():
            t.append([len(word)/len(i),count])
        else:
            t.append([0,count])
        count+=1
    t.sort(reverse=True)
    return words[t[0][1]]


# In[10]:


def make_matrix_cosine(data,song,number):
    df=pd.DataFrame()
    data.drop_duplicates(inplace=True)
    songs=data['song_name'].values
#    best = difflib.get_close_matches(song,songs,1)[0]
    best=find_word(song,songs)
    print('The song closest to your search is :',best)
    genre=data[data['song_name']==best]['genre'].values[0]
    df=data[data['genre']==genre]
    x = df[df['song_name'] == best].drop(columns=['genre', 'song_name']).values
    if len(x) > 0:
        x = x[0]  # Extract the first row as a 1-dimensional array
    else:
        print("No matching song data found.")
        return
    song_names=df['song_name'].values
    df.drop(columns=['genre','song_name'],inplace=True)
    df=df.fillna(df.mean())
    p=[]
    count=0
    for i in df.values:
        p.append([distance.cosine(x,i),count])
        count+=1
    p.sort()
    #for i in range(1,number+1):
        #print(song_names[p[i][1]])
    recommendations = []
    for i in range(1, number + 1):
        recommendations.append(song_names[p[i][1]])

    return recommendations


# In[11]:


def main():
    st.title("Spotify Song Recommendation System")
    
    # Load data from the CSV file
    data = pd.read_csv('genres_v2.csv')

    # Preprocessing steps
    data = data.dropna(subset=['song_name'])
    df = data[data.columns[:11]]
    df['genre'] = data['genre']
    df['time_signature'] = data['time_signature']
    df['duration_ms'] = data['duration_ms']
    df['song_name'] = data['song_name']
    x = df[df.drop(columns=['song_name', 'genre']).columns].values
    scaler = StandardScaler().fit(x)
    X_scaled = scaler.transform(x)
    df[df.drop(columns=['song_name', 'genre']).columns] = X_scaled


    c = st.text_input("Enter the name of the song:")
    d = st.number_input("Enter the number of recommendations you want:", min_value=1, value=5)

    if st.button("Get Recommendations"):
        with st.spinner("Getting recommendations..."):
            recommendations = make_matrix_cosine(df, c, d)
            
        st.subheader(f"Top {d} Song Recommendations:")
        for i, song_name in enumerate(recommendations, start=1):
            st.write(f"{i}. {song_name}")

if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:





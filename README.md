```python
#THIS IS A DETAILED NOTEBOOK OF THE APP.PY PROJECT FILE PRESENT IN THE SAME REPOSITORY
```


```python
#IMPORTING DEPENDENCIES
```


```python
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
import copy
import warnings
warnings.filterwarnings("ignore")
plotly.offline.init_notebook_mode (connected = True)
```


<script type="text/javascript">
window.PlotlyConfig = {MathJaxConfig: 'local'};
if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
if (typeof require !== 'undefined') {
require.undef("plotly");
requirejs.config({
    paths: {
        'plotly': ['https://cdn.plot.ly/plotly-2.12.1.min']
    }
});
require(['plotly'], function(Plotly) {
    window._Plotly = Plotly;
});
}
</script>



# Getting the Data


```python
data=pd.read_csv('genres_v2.csv')
```

# Having First Look At The Data

There are multiple genres in the CSV


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>danceability</th>
      <th>energy</th>
      <th>key</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>...</th>
      <th>id</th>
      <th>uri</th>
      <th>track_href</th>
      <th>analysis_url</th>
      <th>duration_ms</th>
      <th>time_signature</th>
      <th>genre</th>
      <th>song_name</th>
      <th>Unnamed: 0</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.831</td>
      <td>0.814</td>
      <td>2</td>
      <td>-7.364</td>
      <td>1</td>
      <td>0.4200</td>
      <td>0.0598</td>
      <td>0.013400</td>
      <td>0.0556</td>
      <td>0.3890</td>
      <td>...</td>
      <td>2Vc6NJ9PW9gD9q343XFRKx</td>
      <td>spotify:track:2Vc6NJ9PW9gD9q343XFRKx</td>
      <td>https://api.spotify.com/v1/tracks/2Vc6NJ9PW9gD...</td>
      <td>https://api.spotify.com/v1/audio-analysis/2Vc6...</td>
      <td>124539</td>
      <td>4</td>
      <td>Dark Trap</td>
      <td>Mercury: Retrograde</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.719</td>
      <td>0.493</td>
      <td>8</td>
      <td>-7.230</td>
      <td>1</td>
      <td>0.0794</td>
      <td>0.4010</td>
      <td>0.000000</td>
      <td>0.1180</td>
      <td>0.1240</td>
      <td>...</td>
      <td>7pgJBLVz5VmnL7uGHmRj6p</td>
      <td>spotify:track:7pgJBLVz5VmnL7uGHmRj6p</td>
      <td>https://api.spotify.com/v1/tracks/7pgJBLVz5Vmn...</td>
      <td>https://api.spotify.com/v1/audio-analysis/7pgJ...</td>
      <td>224427</td>
      <td>4</td>
      <td>Dark Trap</td>
      <td>Pathology</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.850</td>
      <td>0.893</td>
      <td>5</td>
      <td>-4.783</td>
      <td>1</td>
      <td>0.0623</td>
      <td>0.0138</td>
      <td>0.000004</td>
      <td>0.3720</td>
      <td>0.0391</td>
      <td>...</td>
      <td>0vSWgAlfpye0WCGeNmuNhy</td>
      <td>spotify:track:0vSWgAlfpye0WCGeNmuNhy</td>
      <td>https://api.spotify.com/v1/tracks/0vSWgAlfpye0...</td>
      <td>https://api.spotify.com/v1/audio-analysis/0vSW...</td>
      <td>98821</td>
      <td>4</td>
      <td>Dark Trap</td>
      <td>Symbiote</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.476</td>
      <td>0.781</td>
      <td>0</td>
      <td>-4.710</td>
      <td>1</td>
      <td>0.1030</td>
      <td>0.0237</td>
      <td>0.000000</td>
      <td>0.1140</td>
      <td>0.1750</td>
      <td>...</td>
      <td>0VSXnJqQkwuH2ei1nOQ1nu</td>
      <td>spotify:track:0VSXnJqQkwuH2ei1nOQ1nu</td>
      <td>https://api.spotify.com/v1/tracks/0VSXnJqQkwuH...</td>
      <td>https://api.spotify.com/v1/audio-analysis/0VSX...</td>
      <td>123661</td>
      <td>3</td>
      <td>Dark Trap</td>
      <td>ProductOfDrugs (Prod. The Virus and Antidote)</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.798</td>
      <td>0.624</td>
      <td>2</td>
      <td>-7.668</td>
      <td>1</td>
      <td>0.2930</td>
      <td>0.2170</td>
      <td>0.000000</td>
      <td>0.1660</td>
      <td>0.5910</td>
      <td>...</td>
      <td>4jCeguq9rMTlbMmPHuO7S3</td>
      <td>spotify:track:4jCeguq9rMTlbMmPHuO7S3</td>
      <td>https://api.spotify.com/v1/tracks/4jCeguq9rMTl...</td>
      <td>https://api.spotify.com/v1/audio-analysis/4jCe...</td>
      <td>123298</td>
      <td>4</td>
      <td>Dark Trap</td>
      <td>Venom</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>



# The average time of a song


```python
px.box(data_frame=data,y='duration_ms',color='genre')
```


![newplot (1)](https://github.com/Maxguy0218/Spotify-Song-Recommender-System/assets/118455375/63c1c14a-02d4-4c69-a98b-ef67f2057971)


Well we can clearly see that most of the genres have their own time ranges pystrance is mostly longer.

# HeatMap of the data


```python
data.drop('Unnamed: 0',axis=1,inplace=True)
```


```python
x=list(data.corr().columns)
y=list(data.corr().index)
values=np.array(data.corr().values)
fig = go.Figure(data=go.Heatmap(
    z=values,
    x=x,
    y=y,
                   
    
                   hoverongaps = False))
fig.show()
```
![Heatmap](https://github.com/Maxguy0218/Spotify-Song-Recommender-System/assets/118455375/82902b1b-d468-4720-9911-c7fbef3dcdb3)



```python

# Removeing all the rows with no song name



data=data.dropna(subset=['song_name'])
```

# Preprocessing the Data


```python
# Creating a new dataframe with required features
df=data[data.columns[:11]]
df['genre']=data['genre']
df['time_signature']=data['time_signature']
df['duration_ms']=data['duration_ms']
df['song_name']=data['song_name']
```


```python
x=df[df.drop(columns=['song_name','genre']).columns].values
scaler = StandardScaler().fit(x)
X_scaled = scaler.transform(x)
df[df.drop(columns=['song_name','genre']).columns]=X_scaled
```

# Recommendation System Using Cosine Similarity Distance


```python
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
```


```python
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
    for i in range(1,number+1):
        print(song_names[p[i][1]])
```


```python
c=input('Please enter The name of the song :')
d=int(input('Please enter the number of recommendations you want: '))
make_matrix_cosine(df,c,d)
```

    Please enter The name of the song :bad blood
    Please enter the number of recommendations you want: 5
    The song closest to your search is : Bad Blood
    Heartless
    Lonely
    Flex (feat. Juice WRLD)
    GOTTI
    Doomsday
    

# Thank you

Hope you like it.
This is a detailed notebook of the spotify song recommendation system.
To have a look of the python file that has been hosted, pleases have a look of app.py file.
Cheers!


```python

```

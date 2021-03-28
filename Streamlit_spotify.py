#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 15:26:08 2021

@author: kellylam
"""

import streamlit as st
import base64
import requests
import spotipy
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit.components.v1 as components


# GETTING USER INPUT 
st.title(""" Spotify Optimal Flow Playlist  """)
st.write('Over the years of using Spotify, I have realized that my least favorite function is the shuffle button. The best part of curating your own playlist is being able to order it by how well each track flows into the other and maintaining a thematic environment; however, finding the best way to reorder a playlist and doing so takes a tremendous amount of time (especially with playlists of 100 songs). With quick research, I found that Spotify has audio features for each track which could be used to analyze my favorite songs and essentially reorder a playlist based on.')
user_id = st.text_input("Please enter your Spotify username: ")
playlist_id= st.text_input("Please enter the Spotify playlist uri you wish to reorder. To find your playlist uri, please go to your Spotify and to the playlist you wish to use. Then click the button with three dots that signify 'more info'. Now hover your cursor over share and select 'Copy Spotify URI'. You'll get something similar to spotify:playlist:1TPrIjZTwk2QV2BfuOJvjN. Please enter what comes after 'spotify:playlist:' : ")
############################################################


# Shows playlist
components.iframe(src="https://open.spotify.com/embed/playlist/"+playlist_id, width=700, height=380, scrolling=False)


# from spotipy.oauth2 import SpotifyClientCredentials

import sys
# backend getting tokens
client_id = config.client_id
client_secret = config.client_secret


client_creds = f'{client_id}:{client_secret}'
client_creds_b64 = base64.b64encode(client_creds.encode()) #encode it to bytes to pass through header

token_url = 'https://accounts.spotify.com/api/token'
token_data = {
    'grant_type': 'client_credentials'
}
token_headers = {
    'Authorization': f'Basic {client_creds_b64.decode()}' # decode byte str
}

# Extracting token data to actually get access token
r = requests.post(token_url, data=token_data, headers=token_headers)
data = r.json()
access_token = data['access_token']
# Access token
sp = spotipy.Spotify(auth = access_token)
############################################################


# Get audio features of all the songs in the playlist 
def getPlaylistAudioFeatures(sp, user_id, playlist_id):
    songs = []
    ids = []

    # Based on Spotipy's documentation
    content = sp.user_playlist_tracks(user_id, playlist_id)
    songs += content['items']
              
    # Getting the track ids
    for i in songs:
        ids.append(i['track']['id'])
        
    # Getting the audio feature of each song in json
    index = 0
    audio_features = []
    while index < len(ids):
        # Based on Spotiypy's documentation
        audio_features += sp.audio_features(ids[index:index + 50])
        index += 50
    
    # Formatting
    features_list = []
    for features in audio_features:
        features_list.append([features['energy'], features['liveness'],
                              features['tempo'], features['speechiness'],
                              features['acousticness'], features['instrumentalness'],
                              features['time_signature'], features['danceability'],
                              features['key'], features['duration_ms'],
                              features['loudness'], features['valence'],
                              features['mode'], features['type'],
                              features['uri']])

    # Making the list into a dataframe
    df = pd.DataFrame(features_list, columns = ['energy', 'liveness',
                                                'tempo', 'speechiness',
                                                'acousticness', 'instrumentalness',
                                                'time_signature', 'danceability',
                                                'key', 'duration_ms', 'loudness',
                                                'valence', 'mode', 'type', 'uri'])
    # Export to CSV to merge for future use.
    # df.to_csv(f'{user_id}-{playlist_id}.csv', index=False)
    return df


# Display track name and artist name of the playlist
def showTracks(user_id, playlist_id):
    artist = []
    song = []
    content = sp.user_playlist_tracks(user_id, playlist_id)
    for i, item in enumerate(content['items']):
        track = item['track']
        artist.append(track['artists'][0]['name'])
        song.append(track['name'])
    tracklist = list(zip(song, artist))
    return pd.DataFrame(tracklist, columns=['Song', 'Artist'])


# Get artist ID from each track 
def getArtistID(user_id, playlist_id):
    songs = []
    artist_ids = []
    content = sp.user_playlist_tracks(user_id, playlist_id)
    songs += content['items']
    # Similar method to getting track ID, but instead we get artist ID
    for i in songs:
        artist_ids.append(i['track']['artists'][0]['id'])
    return artist_ids


# Get genre of each track.
def getGenre(artist_ids):
    ids = []
    for i in artist_ids:
        item = sp.artist(i) # Spotify's function
        ids.append(item['genres'][0])
    return pd.DataFrame(ids) 




# Using the functions
audioF = getPlaylistAudioFeatures(sp, user_id, playlist_id)
track_names = showTracks(user_id, playlist_id)
artist_ids = getArtistID(user_id, playlist_id)
genres = getGenre(artist_ids)
# Merging DFs
audioF['Genre'] = genres
audioF['Song'] = track_names['Song']
audioF['Artist'] = track_names['Artist']
# Reorder DF
df = audioF[['Song', 'Artist', 'Genre', 'energy', 'liveness', 'tempo', 'speechiness', 'acousticness', 'instrumentalness', 'danceability', 'key', 'loudness', 'valence', 'mode', 'uri']]

# Show DF
st.write(df)
st.markdown("""Spotify describes the audio features as such: 

* **Energy:**  Represents intensity and activity. Features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.

* **Liveness:** Represents proability of whether or not the track was produced with a live audience. 

* **Tempo:** Beats per minute of the song.
 
* **Speechiness:** Detects the presence of spoken word. The higher the number, the more likely the track is to be similar to an audio book; the lesser the number, the more instrumental it is. 
 
* **Acousticness:** Represents the level of acousticness i.e. not much electronic sounds.

* **Instrumentalness:** Predicts the level of vocalness. No vocal content like rapping or singing will give it a score closer to 0.

* **Danceability:** Describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity.

* **Key:** The musical key of the track. 0 = C, 1 = C♯/D♭, etc. 

* **Loudness:** The avgerage loudness of a track in decibels.

* **Valence:** The musical positiveness conveyed by a track. Tracks with high valence sound more happy and tracks with low valence are sad.

* **Mode:** Indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.""")

from matplotlib.backends.backend_agg import RendererAgg
import matplotlib
_lock = RendererAgg.lock
matplotlib.use("Agg")

row1_1, row1_2 = st.beta_columns(
    (1,1.2))
with row1_1, _lock:
    fig = plt.figure()
    ax = sns.countplot(y = 'Genre', data = df, order = df.Genre.value_counts().index, palette = 'husl')
    plt.title('Distribution of Genre', fontsize = 20)
    plt.xlabel('Number of Tracks', fontsize = 15)
    plt.ylabel('Genre', fontsize = 15)
    # Showing percentages
    total = len(df['Genre'])
    for p in ax.patches:
            percentage = '{:.1f}%'.format(100 * p.get_width()/total)
            x = p.get_x() + p.get_width() + 0.3
            y = p.get_y() + p.get_height() - 0.2
            ax.annotate(percentage, (x, y))
    st.pyplot(fig)

with row1_2, _lock:
    fig = plt.figure()
    ax = sns.countplot(y = 'Artist', data=df, order=df['Artist'].value_counts().index)
    plt.title("Number of Songs to Each Artist", fontsize = 20)
    plt.ylabel('Artist', fontsize = 14)
    plt.xlabel('Number of Tracks', fontsize = 14)
    # Showing the percentage
    total = len(df['Artist'])
    for p in ax.patches:
            percentage = '{:.1f}%'.format(100 * p.get_width()/total)
            x = p.get_x() + p.get_width() + 0.03
            y = p.get_y() + p.get_height() - 0.2
            ax.annotate(percentage, (x, y))
    st.pyplot(fig)
    
row2_1, row2_2= st.beta_columns(
    (1.2,1))
with row2_1, _lock:
    # Mapping the key values to key names
    fig = plt.figure()
    key_mapping = {0.0: 'C', 1.0: 'C♯,D♭', 2.0: 'D', 3.0: 'D♯,E♭', 4.0: 'E', 5.0: 'F', 6.0: 'F♯,G♭', 7.0: 'G', 8.0: 'G♯,A♭', 9.0: 'A', 10.0: 'A♯,B♭', 11.0: 'B'}
    df['key'] = df['key'].map(key_mapping)
    sns.countplot(x = 'key', data = df, order = df['key'].value_counts().index, palette='husl')
    plt.title("Count of Song Keys", fontsize = 17)
    plt.xlabel('Key', fontsize = 13)
    plt.ylabel('Number of Tracks', fontsize = 13)
    st.pyplot(fig)

with row2_2, _lock:
    st.write("Mean value for mode:", df['mode'].mean())
    ax = sns.countplot(x='mode', data=df, palette= 'YlGnBu')
    plt.title("Count of Mode", fontsize = 17)
    plt.xlabel('Mode', fontsize = 14)
    plt.ylabel('Number of Tracks', fontsize = 14)
    # Showing the number on top of the barchart
    # total = len(df['mode'])
    # for p in ax.patches:
    #         x = p.get_x() + p.get_width()/2
    #         height = p.get_height()
    #         ax.text(x, height + 0.4, height, ha = 'center')
    st.pyplot(fig)
    
    
row3_1, row3_2 = st.beta_columns(
    (1,1))
with row3_1, _lock:
    st.write("Mean value for energy:", df['energy'].mean())
    fig = plt.figure()
    sns.histplot(data=df, x = df['energy'], bins = 50, palette = 'steelblue')
    plt.title("Distribution of Energy", fontsize = 17)
    plt.xlabel('Energy', fontsize = 14)
    plt.ylabel('Number of Tracks', fontsize = 14)
    st.pyplot(fig)

with row3_2, _lock:
    st.write("Mean value for tempo:", df['tempo'].mean())
    fig = plt.figure()
    sns.histplot(data=df, x = df['tempo'], bins = 50, palette = 'steelblue')
    plt.title("Distribution of Tempo", fontsize = 17)
    plt.xlabel('Tempo', fontsize = 14)
    plt.ylabel('Number of Tracks', fontsize = 14)
    st.pyplot(fig)


row4_1, row4_2, = st.beta_columns(
    (1,1.1))
with row4_1, _lock:
    st.write("Mean value for valence:", df['valence'].mean())
    fig = plt.figure()
    sns.histplot(data=df, x = df['valence'], bins = 50, palette = 'steelblue')
    plt.title("Distribution of Valence", fontsize = 17)
    plt.xlabel('Valence', fontsize = 14)
    plt.ylabel('Number of Tracks', fontsize = 14)
    st.pyplot(fig)

with row4_2, _lock:
    st.write("Mean value for Danceability:", df['danceability'].mean())
    fig = plt.figure()
    sns.histplot(data = df, x = df['danceability'], bins = 50, palette = 'steelblue')
    plt.title("Distribution of Danceability", fontsize = 17)
    plt.xlabel('Danceability', fontsize = 14)
    plt.ylabel('Number of Tracks', fontsize = 14)
    st.pyplot(fig)


fig = plt.figure()
sns.heatmap(df.corr(), cmap="RdBu_r", annot=True)
st.pyplot(fig)

row5_1, row5_2, = st.beta_columns(
    (1,1))
with row5_1, _lock:
    # Features we want to analyze
    labels = ['energy', 'liveness', 'speechiness', 'acousticness', 'danceability', 'valence', 'mode']
    # Averaging the data and creating the sections
    stats = df[labels].mean().tolist()
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint = False)
    # Plot closure
    stats = np.concatenate((stats,[stats[0]]))
    angles = np.concatenate((angles,[angles[0]]))
    # Size of the figure
    fig = plt.figure(figsize = (20,20))
    # The internal figure
    ax = fig.add_subplot(221, polar=True)
    ax.plot(angles, stats, 'o-', linewidth = 2, color = 'steelblue')
    ax.fill(angles, stats, alpha = 0.25, facecolor = 'steelblue')
    ax.set_thetagrids(angles * 180/np.pi, labels , fontsize = 13)
    # Internal level ticks
    ax.set_rlabel_position(100)
    plt.yticks([0.2 , 0.4 , 0.6 , 0.8  ], ['0.2','0.4', '0.6', '0.8'], color = 'grey', size = 12)
    plt.ylim(0,1)
    plt.title('Mean Values of Audio Features', fontsize = 17)
    ax.grid(True)
    plt.show()
    st.pyplot(fig)
    
with row5_2, _lock:
    analyze = st.selectbox("Choose a song from the playlist that you'd like to examine for audio features.", df['Song'])
    # Features we want to analyze
    labels = ['energy', 'liveness', 'speechiness', 'acousticness', 'danceability', 'valence', 'mode']
    # Choosing Wrong Places as the song to see
    favorite_song = df.loc[df['Song'] == analyze].reset_index()
    # Selecting only the features is calculable and not the String objects like Artist 
    tem = (favorite_song[labels]).transpose()
    stats = tem[0].values.tolist()
    # Selecting the data values to match labels
    # stats = fs[(fs >=0) & (fs < 1)].tolist()
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    # # Plot closure
    stats = np.concatenate((stats,[stats[0]]))
    angles = np.concatenate((angles,[angles[0]]))
    # Size of the figure
    fig = plt.figure(figsize = (20,20))
    # The internal figure
    ax = fig.add_subplot(221, polar = True)
    ax.plot(angles, stats, 'o-', linewidth=2, color = 'steelblue')
    ax.fill(angles, stats, alpha = 0.25, facecolor = 'steelblue')
    ax.set_thetagrids(angles * 180/np.pi, labels , fontsize = 13)
    # Internal level ticks
    ax.set_rlabel_position(100)
    plt.yticks([0.2 , 0.4 , 0.6 , 0.8  ], ['0.2','0.4', '0.6', '0.8'], color = 'grey', size = 12)
    plt.ylim(0,1)
    plt.title('Audio Features of Favorite Song', fontsize = 17)
    ax.grid(True)
    st.pyplot(fig)


# Round for sorting 
df['energy'] = df['energy'].round(1)
df['tempo'] = df['tempo'].round()
df[['valence', 'loudness', 'danceability']] = df[['valence', 'loudness', 'danceability']].round(2)

# Reordering option
options = st.multiselect(
    'What would you like to order your playlist by? Please click the order you wish the playlist to be sorted. Note that you can select 1 up to all 5 audio features.'
    , ['energy', 'tempo', 'valence', 'loudness', 'danceability'], ['energy'])
optimal = df.sort_values(options)

st.write(optimal)
df = df[['Song', 'Artist', 'uri']]


st.write('In order to automate your new playlist into your account please log in using Spotify here: https://developer.spotify.com/dashboard/. Next https://developer.spotify.com/console/post-playlist-tracks/?playlist_id=&position=&uris= follow this link and click get token from OAuth Token then request token. This is essentially giving Spotify permission to create a playlist for you.')

oauth = st.text_input('Please enter your OAuth Token here: ')
oauth='BQC82b01c0s9cDkkmZ7ul3KNVyYaE37hd3iYgTuzclUFH4hLLB59ohblobY0_EPYC8nnY-SmgICkRnsIS8YwniyY6CFgSR-kugaUmDeryDjm-Jz0FcYKNPw8vnrnka1BXDO0NiUa3WcG_dyVevcchN4ZV-eN1m-c1Q-XhP1fCL1c3WXvR48Q3VBXHocRSlNV4H-MYYoGMXW_EZNRBoP-jCnSWs0HsHgMumgzo6Hxlc-CVEp3yWo1hHfH13zWbVzMQrv0V8_2mcj1qtpvRmyDK8cpSBc'

def create_playlist(user_id, oauth):
    request_body = json.dumps({
        'name': 'streamlit',
        'description': 'Playlist reorderd by optimal musical flow on streamlit',
        'public': True
    })
    query = 'https://api.spotify.com/v1/users/{}/playlists'.format(user_id)
    response = requests.post(
        query,
        data = request_body,
        headers = {
            'Content-Type':'application/json',
            'Authorization':'Bearer {}'.format(oauth)
        }
    )
    response_json = response.json()

    # playlist id
    return response_json['id']


def add_to_playlist(new_playlist):        
    # Populate playlist
    request_data = json.dumps(df.uri.tolist())
    query = 'https://api.spotify.com/v1/playlists/{}/tracks'.format(new_playlist)
    response = requests.post(
        query,
        data = request_data,
        headers = {
            'Content-Type':'application/json',
            'Authorization':'Bearer {}'.format(oauth)
        }
    )
    response_json = response.json()
    return response_json


if st.button('Export'):
    cp = create_playlist(user_id, oauth)
    add_to_playlist(cp)
    st.success("Congrats! We did it! Check out your new and improved playlist on your Spotify!")
    st.balloons()


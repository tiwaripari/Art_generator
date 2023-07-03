from dotenv import load_dotenv
import os
import base64
from requests import post, get
import json
import requests
import spotipy 
import pandas as pd
load_dotenv()

client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")

def get_token():
    auth_string = client_id + ":" + client_secret
    auth_bytes = auth_string.encode("utf-8")
    auth_base64 = str(base64.b64encode(auth_bytes), "utf-8")

    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": "Basic " + auth_base64,
        "Content-Type" : "application/x-www-form-urlencoded" 
    }
    data = {"grant_type": "client_credentials"}
    result = post(url, headers=headers, data=data)
    json_result = json.loads(result.content)
    token = json_result["access_token"]
    return token


# def get_auth_header(token):
#     return {"Authorization": "Bearer " + token}

# def search_for_artist(token, artist_name):
#     url = "https://api.spotify.com/v1/search"
#     headers = get_auth_header(token)
#     query = f"?q={artist_name}&type=artist&limit=1"
    
#     query_url = url  + query
#     result = get(query_url, headers = headers)
#     json_result = json.loads(result.content)["artists"]["items"]

#     if len(json_result) == 0:
#         print("No artist with this name found")
#         return None
#     return json_result[0]

    

#token = get_token()
# print(search_for_artist(token, "Pritam"))


def get_track_id(s_name, token):
# Replace 'SONG_NAME' with the name of the song you want to search for
    song_name = s_name

# Make the API request to search for the song
    headers = {'Authorization': 'Bearer ' + token}
    url = f'https://api.spotify.com/v1/search?q={song_name}&type=track'
    response = requests.get(url, headers=headers)

# Parse the response and extract the track ID
    if response.status_code == 200:
        search_results = response.json()
        tracks = search_results['tracks']['items']
        if len(tracks) > 0:
            first_track = tracks[0]
            track_id = first_track['id']
            #print('Track ID:', track_id)
        else:
            print('No tracks found for the given search query.')
    else:
        print('Error occurred while searching for the song:', response.status_code)
    return track_id


# Replace 'SONG_ID' with the actual ID of the song
#song_id = track_id

# Make the API request
def get_features(song_id, token):
    
    headers = {'Authorization': 'Bearer ' + token}
    audio_features_url = f'https://api.spotify.com/v1/audio-features/{song_id}'
    track_url = f'https://api.spotify.com/v1/tracks/{song_id}'

    audio_features_response = requests.get(audio_features_url, headers=headers)
    track_response = requests.get(track_url, headers=headers)

    features = {}

    if audio_features_response.status_code == 200 and track_response.status_code == 200:
        audio_features = audio_features_response.json()
        track_info = track_response.json()

        popularity = track_info['popularity']
        features['popularity'] = popularity
        length = audio_features['duration_ms']
        features['length'] = length
        danceability = audio_features['danceability']
        features['danceability'] = danceability
        acousticness = audio_features['acousticness']
        features['acousticness'] = acousticness
        energy = audio_features['energy']
        features['energy'] = energy
        instrumentalness = audio_features['instrumentalness']
        features['instrumentalness'] = instrumentalness
        liveness = audio_features['liveness']
        features['liveness'] = liveness
        valence = audio_features['valence']
        features['valence'] = valence
        loudness = audio_features['loudness']
        features['loudness'] = loudness
        speechiness = audio_features['speechiness']
        features['speechiness'] = speechiness
        tempo = audio_features['tempo']
        features['tempo'] = tempo
        key = audio_features['key']
        features['key'] = key
        time_signature = audio_features['time_signature']
        features['time_signature'] = time_signature

        

    else:
        return 'Error occurred while retrieving audio features:', audio_features_response.status_code

    df1 = pd.DataFrame.from_dict(features, orient='index').T
    return df1

import requests

# Replace 'YOUR_ACCESS_TOKEN' with the actual access token obtained through authentication
access_token = '90c6a24b1f834ea983552408dff6216c'

# Replace 'SONG_NAME' with the name of the song you want to search for
song_name = 'Kesariya'

# Make the API request to search for the song
headers = {'Authorization': 'Bearer ' + access_token}
url = f'https://api.spotify.com/v1/search?q={song_name}&type=track'
response = requests.get(url, headers=headers)

# Parse the response and extract the track ID
if response.status_code == 200:
    search_results = response.json()
    tracks = search_results['tracks']['items']
    if len(tracks) > 0:
        first_track = tracks[0]
        track_id = first_track['id']
        print('Track ID:', track_id)
    else:
        print('No tracks found for the given search query.')
else:
    print('Error occurred while searching for the song:', response.status_code)

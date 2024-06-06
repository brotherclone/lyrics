import os
import json
import requests
import pandas as pd
from datasets import Dataset


BASE_URL = 'https://www.earthlyframes.com/'
JSON_PATH = "/Users/gabrielwalsh/Sites/lyrics/json"
DATA_PATH = "/Users/gabrielwalsh/Sites/lyrics/data"


def get_json_from_url(url):
    r = requests.get(url)
    return r.json()


def get_song_urls(albums_json):
    song_urls = []
    album_base_url = BASE_URL + 'albums/'
    for album in albums_json:
        album_url = album_base_url + str(album['id'])
        for song in album['songs']:
            song_url = album_url + '/songs/' + str(song['id'])
            song_urls.append((song['title'], song_url, song['id']))
    return song_urls


def save_json_to_file(json_data, file_name, file_path):
    with open(os.path.join(file_path, file_name), 'w') as f:
        json.dump(json_data, f)


def dataset_from_df(data_frame, save_path):
    dataset = Dataset.from_pandas(data_frame)
    dataset.save_to_disk(save_path)


def create_song_name(song_title, song_id):
    letters = [char for char in song_title if char.isalnum()]
    name = ''.join(letters[:5])
    return str(song_id) + '_' + name.lower()


def validate_song_json(song_json):
    required_keys = ['lyrics', 'notes', 'title', 'id', 'album_id', 'song_order', 'trt']
    for key in required_keys:
        if key not in song_json:
            raise ValueError(f"Song JSON does not contain {key}.")
    if song_json['lyrics'] == '':
        return False
    return True


def load_song_data():
    albums_json = get_json_from_url(BASE_URL + '/albums.json')
    save_json_to_file(albums_json, 'albums.json', JSON_PATH+"/albums")
    song_paths = get_song_urls(albums_json)
    for song_title, url, song_id in song_paths:
        song_file_name = create_song_name(song_title, song_id)
        try:
            song_json = get_json_from_url(url + '.json')
            cleaned_json = remove_properties(song_json, ['created_at', 'updated_at', 'streaming_links', 'videos'])
            if validate_song_json(cleaned_json):
                save_json_to_file(cleaned_json, song_file_name + '.json', JSON_PATH+"/songs")
        except json.decoder.JSONDecodeError:
            print(f"Could not load song {song_title} with id {song_id} from {url}")


def remove_properties(json_data, properties_to_remove):
    for prop in properties_to_remove:
        if prop in json_data:
            del json_data[prop]
    return json_data


def validate_data_frame(data_frame):
    required_columns = ['id', 'title', 'lyrics', 'notes', 'album_id', 'song_order', 'trt']
    if not data_frame.columns.is_unique:
        raise ValueError("DataFrame columns are not unique.")
    missing_columns = [column for column in required_columns if column not in data_frame.columns]
    if missing_columns:
        raise ValueError(f"DataFrame is missing required columns: {missing_columns}")
    return True


def load_json_files_from_directory(directory):
    files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.json')]
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
            try:
                dfj = pd.DataFrame([data])
                data_frames.append(dfj)
            except ValueError as e:
                print(e)


if __name__ == '__main__':
    data_frames = []
    load_song_data()
    load_json_files_from_directory(JSON_PATH+"/songs")
    data_frames = pd.concat(data_frames, ignore_index=True)
    dataset_from_df(data_frames, DATA_PATH)

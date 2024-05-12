import getopt
import os
import pandas as pd
import random
import requests
import xmltodict
from parse_yaml import LoadPodcastsFromMarketplace
from typing import List

def get_episodes(obj: LoadPodcastsFromMarketplace) -> List:
  data = requests.get(obj.url)
  feed = xmltodict.parse(data.text)
  episodes = feed["rss"]["channel"]["item"]
  print(f"Found {len(episodes)} episodes.")
  return episodes

def get_random_episode(episodes: List) -> {}:
  return random.choice(episodes)

def download_episode(obj: LoadPodcastsFromMarketplace, episode: {}):
  if episode["enclosure"]["@type"] != obj.file_type:
    print("Only supports audio/mpeg files. Did not download the podcast audio") 
    return

  filename = episode["link"].split('/')[-1] + ".mp3"
  audio_path = os.path.join(obj.episode_folder, filename)

  if not os.path.exists(audio_path):
    print("Downloading title: {0}, from url: {1}".format(episode["title"], episode["link"]))
    audio = requests.get(episode["enclosure"]["@url"])
    with open(audio_path, "wb+") as f:
      f.write(audio.content)
    with open(obj.episode_file_db, "a") as f:
      rec = episode["title"] + "|" + episode["link"] + "|" + filename
      rec += "\n"
      f.write(rec)
  else:
    print("File already exists. Did not download the podcast audio")

def load_downloads_into_pandas(obj: LoadPodcastsFromMarketplace) -> pd.DataFrame:
  col_names = ["title", "link", "filename"]
  df = pd.read_csv(obj.episode_file_db, names=col_names, header=None, sep="|") 
  return df

def load_podcasts_from_marketplace(obj: LoadPodcastsFromMarketplace):
  try:
    episodes: List = get_episodes(obj)
    max_download: int = min(len(episodes), obj.max_download)
    for i in range(max_download):
      episode: {} = get_random_episode(episodes)
      download_episode(obj, episode)
         
    df = load_downloads_into_pandas(obj)
    print(df[["filename", "title"]])
  
  except ValueError as err:
    show_help()


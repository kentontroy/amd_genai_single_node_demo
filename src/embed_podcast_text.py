from get_embedding import get_embedding
from query_chroma_db import show_vector_store_info, show_random_document, show_query_collection
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings.embeddings import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from parse_yaml import EmbedPodcastText
from suppress_std_out import SuppressStdout
from typing import List, Mapping
import json
import math
import numpy as np
import os
import torch
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#########################################################
# Added for Linux support issue
# ChromaDB uses sqllite
try:
  __import__('pysqlite3')
  sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
  pass

import chromadb
#########################################################

def show_statistics(chunks: Mapping[str, str]): 
  word_counts = np.array(list(map(lambda x: len(x.split()), chunks.values()))) 
  quartiles = np.percentile(word_counts, [25, 50, 75])
  print("\nStatistics for podcast text conversion")
  print(f"Word count distribution: {len(word_counts)}") 
  print("Min: %.3f" % word_counts.min())
  print("Min: %.3f" % word_counts.min())
  print("Q1: %.3f" % quartiles[0])
  print("Median: %.3f" % quartiles[1])
  print("Q3: %.3f" % quartiles[2])
  print("Max: %.3f\n" % word_counts.max())

####################################################################################
# According to OpenAI, a token is approximately == 0.75 word
# https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them#
####################################################################################

  token_counts = np.array(list(map(lambda x: math.ceil(float(x) * 0.75), word_counts))) 
  quartiles = np.percentile(token_counts, [25, 50, 75])
  print(f"Token count distribution: {len(token_counts)}") 
  print("Min: %.3f" % token_counts.min())
  print("Min: %.3f" % token_counts.min())
  print("Q1: %.3f" % quartiles[0])
  print("Median: %.3f" % quartiles[1])
  print("Q3: %.3f" % quartiles[2])
  print("Max: %.3f\n" % token_counts.max())

def embed_podcast_text(obj: EmbedPodcastText, show_info=True):
  chunks = chunk_podcast_text(obj)
  vector_store = get_vector_store(obj, chunks)
  if show_info:
    show_random_document(vector_store)
    show_statistics(chunks)
    show_query_collection(vector_store, obj.vector_store_collection, obj.vector_store_query_example)

def chunk_podcast_text(obj: EmbedPodcastText) -> Mapping[str, str]:
  parsed_audio_data: Mapping[str, str] = {}
  for f in os.listdir(obj.episode_folder):
    file_text = ""
    if f.endswith(".txt"):
      if ".DS_S.txt" in f:
        continue
      path_text = os.path.join(obj.episode_folder, f)
      with open(path_text, "r") as f:
        data = json.load(f) 
        for d in data["chunks"]:
          file_text = file_text + d["text"] 
      parsed_audio_data[path_text] = file_text
    else:
      continue 
  return parsed_audio_data 

def get_vector_store(obj: EmbedPodcastText, chunks: Mapping[str, str]) -> Chroma:
  ef: Embedding = get_embedding(obj)  
  chroma_client = chromadb.PersistentClient(path = obj.vector_store_path)
  try:
    collection = chroma_client.get_collection(name=obj.vector_store_collection, embedding_function=ef)
  except ValueError as error:
    collection = chroma_client.create_collection(
      name=obj.vector_store_collection,
      metadata={"hnsw:space": obj.vector_store_hnsw_search_algo} 
    )
    for (k, v) in chunks.items():
      text_splitter = RecursiveCharacterTextSplitter(chunk_size=obj.chunk_size, chunk_overlap=obj.chunk_overlap)
      text_docs = text_splitter.create_documents([v])
      all_splits = text_splitter.split_documents(text_docs)
      all_splits = list(map(lambda x: str(x), all_splits))
      ids = [f"{i}-{k}" for i in range(0, len(all_splits))]
      collection.add(ids=ids, documents=all_splits)
  with SuppressStdout():
    vectorstore = Chroma(
      client = chroma_client,
      collection_name = obj.vector_store_collection,
      embedding_function = ef
    )
    return vectorstore

from langchain_community.vectorstores import Chroma
from langchain_core.embeddings.embeddings import Embeddings
from parse_yaml import EmbedPodcastText
import os
import random
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def show_vector_store_info(vector_store: Chroma):
  keys = vector_store.get().keys()  
  print(keys)
  print(vector_store.get()["ids"])
  print("")

def show_random_document(vector_store: Chroma):
  i = random.randint(0, len(vector_store.get()["documents"]))
  id = vector_store.get()["ids"][i]
  data = vector_store.get()["documents"][i] 
  print(f"id={id}\n")
  print(data)

def query_collection(
  vector_store: Chroma, 
  collection_name: str, 
  query_string: str, 
  n_results=2) -> {}:
  collection = vector_store._client.get_collection(name=collection_name)
  results = collection.query(
    query_texts=[query_string],
    n_results=n_results
  )
  return results  


def show_query_collection(
  vector_store: Chroma, 
  collection_name: str, 
  query_string: str, 
  n_results=2):

  results = query_collection(vector_store, collection_name, query_string, n_results)
  print(results)

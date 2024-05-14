from langchain_community.embeddings import GPT4AllEmbeddings, HuggingFaceEmbeddings
from langchain_core.embeddings.embeddings import Embeddings
import torch

def get_embedding(obj) -> Embeddings:
  if obj.device == "mps":
    device = obj.device if torch.backends.mps.is_available() else "cpu"
  elif obj.device == "cuda": 
    device = obj.device if torch.backends.cuda.is_available() else "cpu"
  else:
    device = "cpu" 
  model_name = obj.vector_store_embedding_model
  model_kwargs = {"device": device}
  encode_kwargs = {"normalize_embeddings": False}
  if model_name.lower() == "gpt4all":
    ef = GPT4AllEmbeddings()
  else:
    ef = HuggingFaceEmbeddings(
      model_name=model_name,
      model_kwargs=model_kwargs,
      encode_kwargs=encode_kwargs
    )
  return ef

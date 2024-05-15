import yaml
from pydantic import BaseModel

class ChatWithRagPDF(BaseModel):
  workflow_type: str = "chat_rag_with_pdf"
  model: str = "mistral:latest"
  device: str = "cpu"
  temperature: float = 0.7
  max_tokens: int = 50
  prompt_template: str = ""
  format_color_of_response: str = ""
  console_line_length: int = 50
  vector_store: str = "chromadb"
  vector_store_path: str = ""
  vector_store_collection: str = "demo"
  vector_store_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
  vector_store_hnsw_search_algo: str = "cosine"
  chunk_size: int = 1000
  chunk_overlap: int = 0
  pdf: str = ""
  speaker_output_action: str = "./src/audio_output.sh"
  speaker_output_voice: str =  "female_03.wav"

class ChatWithRagPodcasts(BaseModel):
  workflow_type: str = "chat_rag_with_podcasts"
  model: str = "mistral:latest"
  device: str = "cpu"
  temperature: float = 0.7
  max_tokens: int = 50
  prompt_template: str = ""
  format_color_of_response: str = ""
  console_line_length: int = 50
  vector_store: str = "chromadb"
  vector_store_path: str = ""
  vector_store_collection: str = "demo"
  vector_store_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
  vector_store_hnsw_search_algo: str = "cosine"
  chunk_size: int = 1000
  chunk_overlap: int = 0
  episode_folder: str = ""
  speaker_output_action: str = "./src/audio_output.sh"
  speaker_output_voice: str =  "female_03.wav"

class EmbedPodcastText(BaseModel):
  workflow_type: str = "embed_podcast_text"
  model: str = "llama2-uncensored:latest"
  device: str = "cpu"
  temperature: float = 0.7
  max_tokens: int = 50
  episode_folder: str = "./audio/podcasts"
  vector_store: str = "chromadb"
  vector_store_path: str = ""
  vector_store_collection: str = "summaries_marketplace_podcasts"
  vector_store_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
  vector_store_hnsw_search_algo: str = "cosine"
  vector_store_query_example: str = "Talk to me about the meaning of life"
  chunk_size: int = 1000
  chunk_overlap: int = 0

class ConvertSpeechToText(BaseModel):
  workflow_type: str = "convert_speech_to_text" 
  model: str = "openai/whisper-large-v3"
  device: str = "cpu"
  episode_folder: str = "./audio/podcasts"
  batch_size: str = "4"
  command: str = "insanely-fast-whisper"

class LoadPodcastsFromMarketplace(BaseModel):
  workflow_type: str = "load_podcasts_from_marketplace"
  url: str = "https://www.marketplace.org/feed/podcast/marketplace/"
  file_type: str = "audio/mpeg"
  max_download: int = 75
  episode_file_db: str = "./podcast_file_db"
  episode_folder: str = "./audio/podcasts"

def process_yaml(filePath: str) -> BaseModel:
  with open(filePath, "r") as file:
    data = yaml.safe_load(file)
  output = {}
  output["workflow_type"] = data["llm_inference"]["workflow_type"]
  if output["workflow_type"] == "chat_rag_with_pdf":
    output["model"] = data["llm_inference"]["model"]["name"]
    output["device"] = data["llm_inference"]["model"]["device"]
    output["temperature"] = data["llm_inference"]["model"]["temperature"]
    output["max_tokens"] = data["llm_inference"]["model"]["max_tokens"]
    output["vector_store"] = data["llm_inference"]["rag"]["vector_store"]
    output["vector_store_path"] = data["llm_inference"]["rag"]["vector_store_path"]
    output["vector_store_collection"] = data["llm_inference"]["rag"]["vector_store_collection"]
    output["vector_store_embedding_model"] = data["llm_inference"]["rag"]["vector_store_embedding_model"]
    output["vector_store_hnsw_search_algo"] = data["llm_inference"]["rag"]["vector_store_hnsw_search_algo"]
    output["chunk_size"] = data["llm_inference"]["rag"]["chunk_size"]
    output["chunk_overlap"] = data["llm_inference"]["rag"]["chunk_overlap"]
    output["pdf"] = data["llm_inference"]["rag"]["pdf"]
    output["format_color_of_response"] = data["llm_inference"]["chat"]["format_color_of_response"]
    output["console_line_length"] = data["llm_inference"]["chat"]["console_line_length"]
    output["prompt_template"] = data["llm_inference"]["chat"]["prompt_template"]
    output["speaker_output_action"] = data["llm_inference"]["processing"]["speaker_output_action"]
    output["speaker_output_voice"] = data["llm_inference"]["processing"]["speaker_output_voice"]
    return ChatWithRagPDF(**output)
  elif output["workflow_type"] == "chat_rag_with_podcasts": 
    output["model"] = data["llm_inference"]["model"]["name"]
    output["device"] = data["llm_inference"]["model"]["device"]
    output["temperature"] = data["llm_inference"]["model"]["temperature"]
    output["max_tokens"] = data["llm_inference"]["model"]["max_tokens"]
    output["vector_store"] = data["llm_inference"]["rag"]["vector_store"]
    output["vector_store_path"] = data["llm_inference"]["rag"]["vector_store_path"]
    output["vector_store_collection"] = data["llm_inference"]["rag"]["vector_store_collection"]
    output["vector_store_embedding_model"] = data["llm_inference"]["rag"]["vector_store_embedding_model"]
    output["vector_store_hnsw_search_algo"] = data["llm_inference"]["rag"]["vector_store_hnsw_search_algo"]
    output["chunk_size"] = data["llm_inference"]["rag"]["chunk_size"]
    output["chunk_overlap"] = data["llm_inference"]["rag"]["chunk_overlap"]
    output["format_color_of_response"] = data["llm_inference"]["chat"]["format_color_of_response"]
    output["console_line_length"] = data["llm_inference"]["chat"]["console_line_length"]
    output["prompt_template"] = data["llm_inference"]["chat"]["prompt_template"]
    output["episode_folder"] = data["llm_inference"]["processing"]["episode_folder"]
    output["speaker_output_action"] = data["llm_inference"]["processing"]["speaker_output_action"]
    output["speaker_output_voice"] = data["llm_inference"]["processing"]["speaker_output_voice"]
    return ChatWithRagPodcasts(**output)
  elif output["workflow_type"] == "embed_podcast_text":
    output["model"] = data["llm_inference"]["model"]["name"]
    output["device"] = data["llm_inference"]["model"]["device"]
    output["temperature"] = data["llm_inference"]["model"]["temperature"]
    output["max_tokens"] = data["llm_inference"]["model"]["max_tokens"]
    output["episoder_folder"] = data["llm_inference"]["processing"]["episode_folder"]
    output["vector_store"] = data["llm_inference"]["vector_store"]["type"]
    output["vector_store_path"] = data["llm_inference"]["vector_store"]["vector_store_path"]
    output["vector_store_collection"] = data["llm_inference"]["vector_store"]["vector_store_collection"]
    output["vector_store_embedding_model"] = data["llm_inference"]["vector_store"]["vector_store_embedding_model"]
    output["vector_store_hnsw_search_algo"] = data["llm_inference"]["vector_store"]["vector_store_hnsw_search_algo"]
    output["vector_store_query_example"] = data["llm_inference"]["vector_store"]["vector_store_query_example"]
    output["chunk_size"] = data["llm_inference"]["vector_store"]["chunk_size"]
    output["chunk_overlap"] = data["llm_inference"]["vector_store"]["chunk_overlap"]
    return EmbedPodcastText(**output)    
  elif output["workflow_type"] == "convert_speech_to_text":
    output["model"] = data["llm_inference"]["model"]["name"]
    output["device"] = data["llm_inference"]["model"]["device"]
    output["episoder_folder"] = data["llm_inference"]["processing"]["episode_folder"]
    output["batch_size"] = data["llm_inference"]["processing"]["batch_size"]
    output["command"] = data["llm_inference"]["processing"]["command"]
    return ConvertSpeechToText(**output)
  elif output["workflow_type"] == "load_podcasts_from_marketplace":
    output["url"] = data["llm_inference"]["url"]
    output["file_type"] = data["llm_inference"]["file_type"]
    output["max_download"] = data["llm_inference"]["max_download"]
    output["episode_file_db"] = data["llm_inference"]["episode_file_db"]
    output["episode_folder"] = data["llm_inference"]["episode_folder"]
    return LoadPodcastsFromMarketplace(**output)

  return null


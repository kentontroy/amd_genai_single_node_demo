import yaml
from pydantic import BaseModel

class ChatWithRagPDF(BaseModel):
  workflow_type: str = ""
  model: str = "mistral:latest"
  device: str = "cpu"
  temperature: float = 0.7
  max_tokens: int = 50
  prompt_template: str = ""
  vector_store: str = "chromadb"
  vector_store_path: str = ""
  vector_store_collection: str = "demo"
  vectore_store_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
  chunk_size: int = 1000
  chunk_overlap: int = 0
  pdf: str = ""

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
    output["vectore_store_embedding_model"] = data["llm_inference"]["rag"]["vectore_store_embedding_model"]
    output["chunk_size"] = data["llm_inference"]["rag"]["chunk_size"]
    output["chunk_overlap"] = data["llm_inference"]["rag"]["chunk_overlap"]
    output["pdf"] = data["llm_inference"]["rag"]["pdf"]
    output["prompt_template"] = data["llm_inference"]["chat"]["prompt_template"]
    return ChatWithRagPDF(**output)
  return null

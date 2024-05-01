from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.embeddings import GPT4AllEmbeddings, HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from parse_yaml import ChatWithRagPDF
from suppress_std_out import SuppressStdout
import chromadb
import torch

def chat_with_rag_pdf(obj: ChatWithRagPDF):
  if obj.device == "mps":
    device = obj.device if torch.backends.mps.is_available() else "cpu"
  elif obj.device == "cuda":
    device = obj.device if torch.backends.cuda.is_available() else "cpu"
  else:
    device = "cpu"
  model_name = obj.vectore_store_embedding_model
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

  chroma_client = chromadb.PersistentClient(path = obj.vector_store_path)
  collection_name = obj.vector_store_collection
  try:
    collection = chroma_client.get_collection(name = collection_name, embedding_function = ef)
  except Exception as error:
    print(error)
    collection = chroma_client.create_collection(name = collection_name)
    loader = PDFMinerLoader(obj.pdf)
    data = loader.load()
    chunk_size = obj.chunk_size
    chunk_overlap = obj.chunk_overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    all_splits = text_splitter.split_documents(data)
    all_splits = list(map(lambda x: str(x), all_splits))
    ids = [str(id) for id in range(0, len(all_splits))]
    collection.add(ids = ids, documents = all_splits)

  with SuppressStdout():
    vectorstore = Chroma(
      client = chroma_client,
      collection_name = collection_name,
      embedding_function = ef
    )
    QA_CHAIN_PROMPT = PromptTemplate(
      input_variables = ["context", "question"],
      template = obj.prompt_template,
    )

  llm = Ollama(model=obj.model,
    temperature = obj.temperature,
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
  )
  qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2}),
    chain_type_kwargs = {"prompt": QA_CHAIN_PROMPT}
  )
#  qa_chain.max_tokens_limit = obj.max_tokens

  while True:
    query = input("\nQuery: ")
    if query == "exit":
      break
    if query.strip() == "":
      continue

    result = qa_chain({"query": query})



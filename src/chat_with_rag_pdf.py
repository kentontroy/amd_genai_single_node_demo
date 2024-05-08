from formatted_console_stream import FormattedConsoleStreamHandler
from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.embeddings import GPT4AllEmbeddings, HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from parse_yaml import ChatWithRagPDF
from suppress_std_out import SuppressStdout
import torch
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#########################################################
# Added for Linux support issue
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
#########################################################
import chromadb

QUERY_HISTORY_STACK = []

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
    callback_manager = CallbackManager(
      [FormattedConsoleStreamHandler(obj.format_color_of_response, obj.console_line_length)])
  )
  qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2}),
    chain_type_kwargs = {"prompt": QA_CHAIN_PROMPT}
  )
##################################################
# TODO: have functionality similar to missing:
#  qa_chain.max_tokens_limit = obj.max_tokens
##################################################

  while True:
    query = input("\nQuery: ")
    if query.strip() == "":
      continue

    if query == "/exit".strip() or query == "exit".strip():
      break
    elif query == "/command".strip() or query == "command".strip():
      command = input("\nCommand: ")
      if command == "last run".strip():
        if len(QUERY_HISTORY_STACK) > 0:
          query = QUERY_HISTORY_STACK[-1]
        else:
          print("No previous LLM invocation can be found.")
          continue
      elif command == "exit".strip():
        continue 
      elif command == "help".strip():
        print("last run:  Invokes the LLM using the last prompt")
        print("last save: Saves the output from the last LLM invocation")
        continue
      else:
        if command not in ["last run"]:
          print("Command mode options include: last run, last save")
          continue
    else:
      QUERY_HISTORY_STACK.append(query)

    result = qa_chain({"query": query})



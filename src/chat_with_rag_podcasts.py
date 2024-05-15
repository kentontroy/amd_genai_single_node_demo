from formatted_console_stream import FormattedConsoleStreamHandler
from get_embedding import get_embedding
from run_console_loop import run_console_loop
from langchain_core.embeddings.embeddings import Embeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.chains import RetrievalQA
from parse_yaml import ChatWithRagPodcasts
from suppress_std_out import SuppressStdout
import subprocess
import sys
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

def chat_with_rag_podcasts(obj: ChatWithRagPodcasts):
  ef: Embedding = get_embedding(obj)  
  chroma_client = chromadb.PersistentClient(path = obj.vector_store_path)
  collection_name = obj.vector_store_collection
  try:
    collection = chroma_client.get_collection(name=collection_name, embedding_function=ef)
  except Exception as error:
    print(error)
    sys.exit(1)

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

  run_console_loop(qa_chain, obj)

from parse_yaml import ChatWithRagPDF, ConvertSpeechToText, process_yaml
from chat_with_rag_pdf import chat_with_rag_pdf
from convert_podcast_to_text import convert_speech_to_text
import getopt
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def show_help():
  print("main.py -y <YAML>")
  sys.exit(1)

if __name__== '__main__':
  c_args = sys.argv[1:]
  if len(c_args) == 0:
    show_help()
  c_options = "y:"
  c_long_options = ["yaml="]
  try:
    args, vals = getopt.getopt(c_args, c_options, c_long_options)
    for arg, val in args:
      if arg in ["-y", "--yaml"]:
        obj = process_yaml(val)
        if isinstance(obj, ChatWithRagPDF):
          chat_with_rag_pdf(obj)
        elif isinstance(obj, ConvertSpeechToText):
          convert_speech_to_text(obj)
      else:
        show_help()

  except getopt.error as err:
    print(str(err))


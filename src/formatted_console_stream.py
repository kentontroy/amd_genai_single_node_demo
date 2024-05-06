from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.outputs import LLMResult
from typing import Any
import sys

FormattedColorMappings = {
  "PINK": "\033[95m", 
  "CYAN": "\033[96m",
  "YELLOW": "\033[93m",
  "NEON_GREEN": "\033[92m",
  "RESET_COLOR": "\033[0m"
}

class FormattedConsoleStreamHandler(StreamingStdOutCallbackHandler):
  def __init__(self, format_color_of_response: str, console_line_length: int):
    if format_color_of_response in FormattedColorMappings:
      self.format_color_of_response = FormattedColorMappings[format_color_of_response.upper()] 
    else:
      self.format_color_of_response = FormattedColorMappings["RESET_COLOR"]
    self.console_line_length = console_line_length
    self.line_buffer = ""

  def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
    self.line_buffer += token
    if len(self.line_buffer) >= self.console_line_length:
      self.line_buffer = self.line_buffer.replace("\n","")
      self.line_buffer = self.line_buffer.replace("\t","")
      self.line_buffer = self.line_buffer.strip()
      sys.stdout.write(self.format_color_of_response)
      sys.stdout.write(self.line_buffer)
      sys.stdout.write(FormattedColorMappings["RESET_COLOR"])
      sys.stdout.write("\n")
      sys.stdout.flush()
      self.line_buffer = ""

  def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None: 
      self.line_buffer = self.line_buffer.replace("\n","")
      self.line_buffer = self.line_buffer.replace("\t","")
      self.line_buffer = self.line_buffer.strip()
      sys.stdout.write(self.format_color_of_response)
      sys.stdout.write(self.line_buffer)
      sys.stdout.write(FormattedColorMappings["RESET_COLOR"])
      sys.stdout.write("\n")
      sys.stdout.flush()
      self.line_buffer = ""

from langchain.chains import RetrievalQA
import subprocess

def run_console_loop(qa_chain: RetrievalQA, obj):
  QUERY_HISTORY_STACK = []
  speak_enabled = False

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
      elif command == "speak on".strip():
        speak_enabled = True
        continue 
      elif command == "speak off".strip():
        speak_enabled = False
        continue 
      elif command == "exit".strip():
        continue 
      elif command == "help".strip():
        print("last run:  Invokes the LLM using the last prompt")
        print("last save: Saves the output from the last LLM invocation")
        print("speak on:  Turns on voice assistant audio")
        print("speak off: Turns off voice assistant audio")
        print("exit: Leave command mode and back to the query prompt")
        continue
      else:
        print("Command mode options include: last run, last save, speak on, speak off, exit")
        continue
    else:
      QUERY_HISTORY_STACK.append(query)

    result = qa_chain({"query": query})

    if speak_enabled:
# Avoid the error: https://errors.pydantic.dev/2.6/v/string_too_long
      n = len(result["result"]) 
      i = n if n <= 2000 else 2000
      subprocess.run([obj.speaker_output_action, result["result"][0:i], obj.speaker_output_voice])

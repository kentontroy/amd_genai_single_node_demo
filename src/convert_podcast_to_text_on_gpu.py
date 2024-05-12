from parse_yaml import ConvertSpeechToText
import os
import subprocess

def convert_speech_to_text(obj: ConvertSpeechToText):
  try:
    audio_files = []
    for (dirpath, dirname, filenames) in os.walk(obj.episode_folder):
      audio_files.extend(filenames) 

########################################################################################
# Example command being formed below:
# pipx run insanely-fast-whisper 
#  --file-name ./audio/podcasts/a-labor-market-paradox.mp3 
#  --device-id mps --model-name openai/whisper-large-v3 
#  --batch-size 4 --transcript-path ./audio/podcasts/a-labor-market-paradox.txt
########################################################################################
      for f in audio_files:
        path_txt = os.path.join(obj.episode_folder, f[:-4] + ".txt")
# If the audio file has already been converted the disregard
        if not os.path.exists(path_txt):
          path_mp3 = os.path.join(obj.episode_folder, f[:-4] + ".mp3")
          subprocess.run([
            "pipx", "run", obj.command, "--file-name", path_mp3, "--device-id", obj.device,
            "--model-name", obj.model, "--batch-size", obj.batch_size, "--transcript-path", path_txt
          ])

  except Exception as err:
    print(str(err))

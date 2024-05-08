from faster_whisper import WhisperModel
import signal
import sys

AUDIO_FILE = "./audio/test/sam_altman_lex_podcast_367.flac"
WHISPER_MODEL_SIZE = "large-v2"
DEVICE = "cpu"

FormattedColorMappings = {
  "PINK": "\033[95m",
  "CYAN": "\033[96m",
  "YELLOW": "\033[93m",
  "NEON_GREEN": "\033[92m",
  "RESET_COLOR": "\033[0m"
}

def signal_handler(sig, frame):
  sys.stdout.write(FormattedColorMappings["RESET_COLOR"])
  sys.stdout.flush()
  sys.exit(0)

if __name__== "__main__":
  model = WhisperModel(WHISPER_MODEL_SIZE, device=DEVICE, compute_type="float32")
  segments, info = model.transcribe(AUDIO_FILE, beam_size=1)
  print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

  signal.signal(signal.SIGINT, signal_handler)
  print("Press Ctrl+C to stop")
  print("--------------------\n")

  i = 0
  for segment in segments:
    if i % 2 == 0:
      sys.stdout.write(FormattedColorMappings["NEON_GREEN"])
    else:
      sys.stdout.write(FormattedColorMappings["CYAN"])
    i += 1
    sys.stdout.write(segment.text)
    sys.stdout.write("\n")
    sys.stdout.flush()

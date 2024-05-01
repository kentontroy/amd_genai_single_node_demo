from faster_whisper import WhisperModel

AUDIO_FILE = "/Users/statisticalfx/Documents/Projects/StatisticalFX/pytorch/textGen/intel_demo/audio/sam_altman_lex_podcast_367.flac"

model_size = "large-v2"
model = WhisperModel(model_size, device="cpu", compute_type="float32")
segments, info = model.transcribe(AUDIO_FILE, beam_size=1)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
     
for segment in segments:
    print(segment.text)

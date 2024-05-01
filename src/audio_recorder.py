from pvrecorder import PvRecorder
from typing import List

if __name__== '__main__':
  devices: List[str] = PvRecorder.get_available_devices()
  print(len(devices))

  for index, device in enumerate(devices):
    print(f"[{index}] {device}")

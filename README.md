# amd_genai_single_node_demo

## Installation instructions for RHEL 8 
```
cat /etc/redhat-release
Red Hat Enterprise Linux release 8.6 (Ootpa)
```
## Install RHEL repos
```
sudo dnf -y install https://download.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm
sudo dnf -y install --nogpgcheck https://mirrors.rpmfusion.org/free/el/rpmfusion-free-release-8.noarch.rpm
sudo dnf -y install --nogpgcheck https://mirrors.rpmfusion.org/nonfree/el/rpmfusion-nonfree-release-8.noarch.rpm
sudo subscription-manager repos --enable codeready-builder-for-rhel-8-x86_64-rpms
```
## Install ffmpeg audio dependencies required for speech use cases
```
sudo dnf install ffmpeg
ffmpeg -version
```
## Install Development tools
```
sudo yum install git
sudo dnf group install "Development Tools"
sudo dnf install llvm-toolset
gcc --version
gcc (GCC) 8.5.0 20210514 (Red Hat 8.5.0-20)
```
## Install additional audio and _ctype dependencies to deal with -devel wheel building
```
rpm -qa | grep libffi
libffi-3.1-23.el8.x86_64

sudo yum install -y libffi-devel bzip2-devel readline-devel libsqlite3x-devel.x86_64
sudo yum install portaudio-devel alsa-lib-devel portaudio
```
## Install and use Python virtual environments
```
sudo curl https://pyenv.run | bash
cat $HOME/.bash_profile
...
# User specific environment and startup programs
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

pyenv install 3.10.0
pyenv install 3.11.9

pyenv virtualenv 3.11.9 venv
pyenv activate venv
```
## Speech-to-text via a GPU - given Metal and Cuda support
```
(venv) [cloudera@mxq3180fg6-os demo]$ python --version
Python 3.11.9

pip --require-virtualenv install -r ./requirements.txt
pipx install insanely-fast-whisper
  installed package insanely-fast-whisper 0.0.8, installed using Python 3.11.9
  These apps are now globally available
    - insanely-fast-whisper
pipx ensurepath
pipx list

cd ./audio/test
wget https://huggingface.co/datasets/reach-vb/random-audios/resolve/main/sam_altman_lex_podcast_367.flac
pipx run insanely-fast-whisper --file-name sam_altman_lex_podcast_367.flac --device-id cpu --model-name openai/whisper-large-v3 --batch-size 4
```
## Text-to-text using Ollama
```
(venv) [cloudera@mxq3180fg6-os demo]$ curl -fsSL https://ollama.com/install.sh | sh
>>> The Ollama API is now available at 127.0.0.1:11434.
>>> Install complete. Run "ollama" from the command line.
WARNING: No NVIDIA/AMD GPU detected. Ollama will run in CPU-only mode.

The Web interface starts automatically upon install. To start it manually,
ollama serve >> logs/ollama-log.txt 2>&1

ollama pull llama2-uncensored:latest
ollama pull llama3
```
## Run an LLM text-to-text example
```
(venv) [cloudera@mxq3180fg6-os demo]$ python src/main.py -y config_chat_pdf.yml
```
## Text-to-speech using AllTalk_tts
```
(venv_alltalk_tts) [cloudera@mxq3180fg6-os demo]$ git clone https://github.com/erew123/alltalk_tts
(venv_alltalk_tts) [cloudera@mxq3180fg6-os demo]$ pip install -r system/requirements/requirements_standalone.txt

(venv_alltalk_tts) [cloudera@mxq3180fg6-os demo]$ python script.py
[AllTalk Startup] Model is available     : Checked
[AllTalk Startup] Current Python Version : 3.11.9
[AllTalk Startup] Current PyTorch Version: 2.3.0
[AllTalk Startup] Current CUDA Version   : CUDA is not available
[AllTalk Startup] Current TTS Version    : 0.22.0
[AllTalk Startup] Current TTS Version is : Up to date
[AllTalk Startup] AllTalk Github updated : 12th May 2024 at 13:00
[AllTalk Startup] TTS Subprocess         : Starting up
[AllTalk Startup]
[AllTalk Startup] AllTalk Settings & Documentation: http://127.0.0.1:7851
```

# **Neural Needledrop:** Pipeline

This folder contains code related to the data pipeline for Neural Needledrop.

---

## Setting up Your Environment

I'm using Poetry to manage my Python dependencies. In order to install them, you can run the following commands.

**Note:** Please run all of the commands from the `neural-needledrop/pipeline` directory.

```
pip install poetry
poetry install --no-root
```

Next, you'll have to run `pip install -r requirements.txt`. This is because I want to download Whisper straight from the OpenAI GitHub repo, but I couldn't quite figure it out for `poetry` yet.

Finally, you'll need to download `ffmpeg`. I'm on Windows right now, so I have to use Chocolatey. From [this guide](https://adamtheautomator.com/install-ffmpeg/#Method_2_Install_FFmpeg_via_Chocolatey), I learned how to install this via Powershell:

```
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
choco install ffmpeg
```

Eventually, I'll take care of this via a `Dockerfile`.

DOCKER RUN:

```
docker build -t neural-needledrop-pipeline .

docker run -it -e PYTHONBUFFERED=1 -e LOG_TO_CONSOLE=True -e TQDM_ENABLED=True -e OPENAI_API_KEY=[INSERT OPENAI API KEY] neural-needledrop-pipeline
```
 
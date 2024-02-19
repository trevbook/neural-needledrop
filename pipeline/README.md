# **Neural Needledrop:** Pipeline

This folder contains code related to the data pipeline for Neural Needledrop. This data pipeline runs through the following steps: 

- **Initialize Cloud Resources:** Make sure that all of the GBQ tables & GCS buckets exist. 

- **Download Video Metadata:** Identify videos that haven't been indexed, and download their metadata. 

- **Enrich Video Metadata:** This step will determine what type of video each video is (album review, weekly track roundup, etc.), and extract review scores from the description

- **Downloading Video Audio:** This step will download the audio of videos we haven't already downloaded yet. 

- **Transcribing Audio:** Next: this step uses [OpenAI's Whisper model](https://github.com/openai/whisper) to transcribe all of the audio we've downloaded. 

- **Embedding Transcriptions:** Finally, we're going to embed some of the transcriptions that we've created using Whisper. We'll use [the embeddings API that OpenAI provides](https://platform.openai.com/docs/guides/embeddings) for this.

---

## Setting up Your Environment (LOCAL)
*Below, I've included some instructions for setting up the environment for local use. You can also just use the Dockerized version of the pipeline, too.*

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

---

## Running the Pipeline (LOCALLY VIA DOCKER)
I've created a Dockerized version of the pipeline that can be run using the following commands. 

***NOTE:*** *These instructions assume you have a GCP service account key in `pipeline/gcloud-service-key.json`. Without this key, the pipeline will not work properly.*

```
# Build the Docker image
docker build -t neural-needledrop-pipeline .

# Run the Docker image
docker run -it -e PYTHONBUFFERED=1 -e LOG_TO_CONSOLE=True -e TQDM_ENABLED=True -e OPENAI_API_KEY=[INSERT OPENAI API KEY] neural-needledrop-pipeline
```
 
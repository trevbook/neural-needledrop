# **Neural Needledrop:** API
This folder contains code related to the API. The API is written in Python, using FastAPI.  

--- 
## Setting up Your Environment
I'm using Poetry to manage my Python dependencies. In order to install them, you can run the following commands. 

**Note:** Please run all of the commands from the `neural-needledrop/api` directory. 

```
pip install poetry
poetry install --no-root
```

---
## Running the API Locally

```
$env:LOG_TO_CONSOLE="True"
$env:LOG_LEVEL="DEBUG"
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

## Docker Image

BUILDING: 
```
docker build -t neural-needledrop-api .
```

RUNNING: 
```
docker run -it -e PYTHONBUFFERED=1 -e POSTGRES_HOST=neural-needledrop-database -e LOG_TO_CONSOLE=True -e TQDM_ENABLED=True -e OPENAI_API_KEY=$OPENAI_API_KEY -p 8000:8000 neural-needledrop-api
```
   
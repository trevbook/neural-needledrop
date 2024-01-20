# **Neural Needledrop**
This is my AI-powered search engine for Anthony Fantano's "TheNeedleDrop" reviews. 

I need to write this README out a little more, but that'll come later. 

--- 
# Webapp Components
This repo is something of a monorepo for the entire application. It contains four main components: 

- **Pipeline:** The data pipeline for Neural Needledrop, which updates my GBQ dataset with information about new NeedleDrop videos as they release. 

- **Database:** The database for Neural Needledrop. This will download relevant data from GBQ and GCS, and then set up the databases necessary for the API to function properly. Right now, I'm planning to create a Postgres database, and *maybe* add on a Solr database later.  

- **API:** The API for the Neural Needledrop webapp. This is a FastAPI-backed API that will work as an interface for the database; it will also power the UI.  

- **UI:** The user interface of the Neural Needledrop webapp. 

Each of these components has been separated into their own folder in the repo. You'll find similar `README` files within, explaining more about each step. 


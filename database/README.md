# **Neural Needledrop:** Database
All of the code in this file pertains to the local databases that will be run on the production machine. 

```
docker run --name neural-needledrop-database -e POSTGRES_PASSWORD=my_password -e POSTGRES_DB=neural_needledrop_data -e POSTGRES_USER=my_user -p 5432:5432 -v ${PWD}/data:/var/lib/postgresql/data -d ankane/pgvector
```

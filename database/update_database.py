"""
This script will update the postgres database with the latest data from the Google BigQuery database.
"""

# =====
# SETUP
# =====
# The code below will set up the rest of the script!

# General import statements
from pandas_gbq import read_gbq
from pathlib import Path
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from sqlalchemy import create_engine, MetaData, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.sql import text
from pgvector.sqlalchemy import Vector
from tqdm import tqdm
import traceback

# Importing modules custom-built for this project
from utils.settings import (
    POSTGRES_USER,
    POSTGRES_PASSWORD,
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_DB,
    LOG_TO_CONSOLE,
    GBQ_PROJECT_ID,
    GBQ_DATASET_ID,
)
from utils.logging import get_logger
from utils.postgres import delete_table, create_table
from utils.gcs import download_file_from_bucket
from utils.postgres import query_postgres, upload_to_table, delete_table

# Set up a logger for this notebook
logger = get_logger("database_update", log_to_console=LOG_TO_CONSOLE)

# Create the connection string to the database
postgres_connection_string = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
logger.info(
    f"Connecting to the database using the following connection string: {postgres_connection_string}"
)

# Create the connection engine
engine = create_engine(postgres_connection_string)
metadata = MetaData()
session = sessionmaker(bind=engine)()
Base = declarative_base()

# ========================
# DATABASE UPDATE FUNCTION
# ========================
# Below, I've functionalized the database update process so that it can be run in a single line of code.


def update_database(
    database_folder="data",
    recreate_tables=False,
    postgres_upload_chunksize=5000,
    postgres_maintenance_work_mem="2GB",
    postgres_max_parallel_maintenance_workers=1,
    max_n_videos_to_update_embeddings=100,
    n_max_embeddings_per_upload=500,
    recreate_embeddings_index_if_exists=True,
    embeddings_index_ivfflat_nlist=500,
):
    """
    This function updates the database with the latest data.

    Parameters:
    database_folder (str): The folder where the data is stored. Default is "data".
    recreate_tables (bool): Whether we want to re-create tables if they already exist. Default is False.
    postgres_upload_chunksize (int): The chunksize for the Postgres upload. Default is 5000.
    postgres_maintenance_work_mem (str): The memory allocated for maintenance operations. Default is "6GB".
    postgres_max_parallel_maintenance_workers (int): The maximum number of workers that can be used for maintenance operations. Default is 7.
    max_n_videos_to_update_embeddings (int): The maximum number of videos to update the embeddings for. Default is 100.
    n_max_embeddings_per_upload (int): The maximum number of embeddings to upload at once. Default is 500.
    recreate_embeddings_index_if_exists (bool): Whether to recreate the embeddings index if it already exists. Default is True.
    embeddings_index_ivfflat_nlist (int): The number of Voronoi cells (i.e., clusters) for the IVFFLAT index. Default is 500.
    """

    # =======================
    # OPTIONAL TABLE DELETION
    # =======================
    # If the user wants to delete the tables, then do so

    if recreate_tables:
        # Log that we're deleting the tables
        logger.info("DELETING EACH OF THE POSTGRES TABLES...")
        tables_to_delete = [
            "video_metadata",
            "embeddings",
            "transcriptions",
            "embeddings_to_text",
        ]
        for table in tables_to_delete:
            delete_table(table, engine, logger)

    # =============================
    # VIDEO_METADATA INITIALIZATION
    # =============================
    # Below, we'll initialize the video_metadata table in the database

    # Log that we're creating the video_metadata table
    logger.info("CREATING THE video_metadata TABLE...")

    try:

        # Define the schema that we'll be using for this table
        schema = [
            Column("id", String, primary_key=True),
            Column("title", String),
            Column("length", Integer),
            Column("channel_id", String),
            Column("channel_name", String),
            Column("short_description", String),
            Column("description", String),
            Column("view_ct", Integer),
            Column("url", String),
            Column("small_thumbnail_url", String),
            Column("large_thumbnail_url", String),
            Column("video_type", String),
            Column("review_score", Integer),
            Column("publish_date", DateTime),
            Column("scrape_date", DateTime),
        ]

        # Create the table
        create_table("video_metadata", schema, engine, metadata, logger)

    except Exception as e:

        # Log the error
        logger.error(f"An error occurred while creating the video_metadata table: {e}")

    # =============================
    # TRANSCRIPTIONS INITIALIZATION
    # =============================
    # Below, we'll initialize the transcriptions table in the database

    # Log that
    logger.info("CREATING THE transcriptions TABLE...")

    try:

        # Define the schema that we'll be using for this table
        transcriptions_table_schema = [
            Column("url", String),
            Column("text", String),
            Column("segment_id", Integer),
            Column("segment_seek", Integer),
            Column("segment_start", Integer),
            Column("segment_end", Integer),
        ]

        # Create the table
        create_table(
            "transcriptions", transcriptions_table_schema, engine, metadata, logger
        )

    except Exception as e:

        # Log the error
        logger.error(f"An error occurred while creating the transcriptions table: {e}")

    # Try and add the ts_vec column if it doesn't exist
    try:

        # Log that we're adding the ts_vec column
        logger.info("Trying to add the ts_vec column to the transcriptions table...")

        # Declare a query that'll identify whether the ts_vec column exists
        tsvector_column_exists_query = """
        SELECT
        column_name
        FROM
        information_schema.columns
        WHERE
        table_name = 'transcriptions'
        AND
        column_name = 'ts_vec';
        """

        # Execute the query
        tsvector_column_exists = (
            query_postgres(tsvector_column_exists_query, engine).shape[0] > 0
        )

        # We're only going to run this notebook if the ts_vec column doesn't exist
        if not tsvector_column_exists:

            # Log that we're adding the ts_vec column
            logger.info("ADDING THE ts_vec COLUMN TO THE transcriptions TABLE...")

            # Define a query that'll add a tsvector column to the transcriptions table
            add_tsvector_column_query = """
            ALTER TABLE transcriptions
            ADD COLUMN ts_vec TSVECTOR
            GENERATED ALWAYS AS (
                to_tsvector('english', text) 
            ) STORED;
            """

            # Execute the query
            query_postgres(add_tsvector_column_query, engine)

    except Exception as e:

        # Log the error
        logger.error(
            f"An error occurred while adding the ts_vec column to the transcriptions table: {e}"
        )

        # Log the traceback
        logger.error(traceback.format_exc())

    # =========================
    # EMBEDDINGS INITIALIZATION
    # =========================
    # Below, we'll initialize the embeddings table in the database

    # Log that
    logger.info("CREATING THE embeddings TABLE...")

    try:
        # Enable the pgvector Extension
        session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        session.commit()

        # Now, we're going to create a table for the embeddings
        embeddings_table_schema = [
            Column("id", String, primary_key=True),
            Column("url", String),
            Column("embedding_type", String),
            Column("start_segment", Integer),
            Column("end_segment", Integer),
            Column("segment_length", Integer),
            Column("embedding", Vector(1536)),
        ]

        # Now, we're going to create a table for the embeddings
        create_table("embeddings", embeddings_table_schema, engine, metadata, logger)

    except Exception as e:
        # Log the error
        logger.error(f"An error occurred while creating the embeddings table: {e}")

    finally:
        # Close the session
        session.close()

    # =============================
    # CREATING TEMPORARY GBQ TABLES
    # =============================
    # Below, I'm going to create temporary tables in GBQ that will allow me to compare the current state of the Postgres database with the state of the GBQ database.
    # This will allow me to determine which videos, transcriptions, and embeddings are currently in the Postgres database, but not in the GBQ database.

    # Log that we're creating the temporary GBQ tables
    logger.info("CREATING TEMPORARY GBQ TABLES...")

    # Determine the videos currently in the `video_metadata` table
    cur_database_video_metadata_df = query_postgres(
        "SELECT id FROM video_metadata",
        engine=engine,
        logger=logger,
    )

    # Determine the transcriptions currently in the `transcriptions` table
    cur_database_transcriptions_df = query_postgres(
        "SELECT DISTINCT(url) FROM transcriptions",
        engine=engine,
        logger=logger,
    )

    # Determine the embeddings currently in the `embeddings` table
    cur_database_embeddings_df = query_postgres(
        "SELECT id, url FROM embeddings GROUP BY id, url",
        engine=engine,
        logger=logger,
    )

    # Upload the `cur_database_video_metadata_df` dataframe to a temporary table in GBQ
    cur_database_video_metadata_df.to_gbq(
        f"{GBQ_PROJECT_ID}.{GBQ_DATASET_ID}.cur_pg_db_video_metadata",
        if_exists="replace",
    )

    # Upload the `cur_database_transcriptions_df` dataframe to a temporary table in GBQ
    cur_database_transcriptions_df.to_gbq(
        f"{GBQ_PROJECT_ID}.{GBQ_DATASET_ID}.cur_pg_db_transcriptions",
        if_exists="replace",
    )

    # Upload the `cur_database_embeddings_df` dataframe to a temporary table in GBQ
    cur_database_embeddings_df.to_gbq(
        f"{GBQ_PROJECT_ID}.{GBQ_DATASET_ID}.cur_pg_db_embeddings",
        if_exists="replace",
    )

    # ====================================
    # DETERMINING DELTA FOR video_metadata
    # ====================================
    # Below, I'm going to determine the delta between the current state of the Postgres database and the state of the GBQ database.

    # Log that we're determining the delta for the video_metadata table
    logger.info("DETERMINING DELTA FOR video_metadata...")

    # Define a query that'll grab all of the video metadata from the GBQ database
    video_metadata_query = f"""
    -- This query will select metadata for all of the videos that have transcriptions & embeddings
    SELECT
    video.*,
    enriched_video.video_type,
    enriched_video.review_score
    FROM
    `{GBQ_PROJECT_ID}.{GBQ_DATASET_ID}.video_metadata` video
    JOIN
    `{GBQ_PROJECT_ID}.{GBQ_DATASET_ID}.enriched_video_metadata` enriched_video
    ON
    video.id = enriched_video.id
    WHERE
    video.url IN (SELECT DISTINCT(url) FROM `{GBQ_PROJECT_ID}.{GBQ_DATASET_ID}.transcriptions`)
    AND
    video.url IN (SELECT DISTINCT(video_url) AS url FROM `{GBQ_PROJECT_ID}.{GBQ_DATASET_ID}.embeddings`)
    AND
    video.id NOT IN (SELECT id FROM `{GBQ_PROJECT_ID}.{GBQ_DATASET_ID}.cur_pg_db_video_metadata`)
    """

    # Execute the above query
    video_metadata_df = read_gbq(video_metadata_query)

    # ====================================
    # DETERMINING DELTA FOR transcriptions
    # ====================================
    # Below, I'm going to determine the delta between the current state of the Postgres database and the state of the GBQ database.

    # Log that we're determining the delta for the transcriptions table
    logger.info("DETERMINING DELTA FOR transcriptions...")

    # Declare the query that will download all of the relevant rows from the
    # transcription table
    transcriptions_query = f"""
    SELECT 
    transcription.url,
    transcription.text,
    transcription.segment_id,
    transcription.segment_seek,
    transcription.segment_start,
    transcription.segment_end
    FROM
    `{GBQ_PROJECT_ID}.{GBQ_DATASET_ID}.transcriptions` transcription
    JOIN (
    SELECT
        video.url
    FROM
        `neural-needledrop.backend_data.video_metadata` video
    WHERE
        video.url IN (SELECT DISTINCT(url) FROM `{GBQ_PROJECT_ID}.{GBQ_DATASET_ID}.transcriptions`)
        AND
        video.url IN (SELECT DISTINCT(video_url) AS url FROM `{GBQ_PROJECT_ID}.{GBQ_DATASET_ID}.embeddings`)
        AND
        video.id NOT IN (SELECT id FROM `{GBQ_PROJECT_ID}.{GBQ_DATASET_ID}.cur_pg_db_video_metadata`)
    ) video
    ON
    video.url = transcription.url
    """

    # Execute the above query
    transcriptions_df = read_gbq(transcriptions_query)

    # ================================
    # DETERMINING DELTA FOR embeddings
    # ================================
    # Below, I'm going to determine the delta between the current state of the Postgres database and the state of the GBQ database.

    # Log that we're determining the delta for the embeddings table
    logger.info("DETERMINING DELTA FOR embeddings...")

    # Declare the query that will download the `embeddings` table
    embeddings_query = f"""
    SELECT 
    embedding.id,
    embedding.video_url AS url,
    embedding.embedding_type,
    embedding.start_segment,
    embedding.end_segment,
    embedding.segment_length,
    embedding.gcs_uri
    FROM
    `{GBQ_PROJECT_ID}.{GBQ_DATASET_ID}.embeddings` embedding
    WHERE
    embedding.video_url IN (
        SELECT DISTINCT(video_url) 
        FROM `{GBQ_PROJECT_ID}.{GBQ_DATASET_ID}.embeddings` 
        WHERE video_url NOT IN (SELECT url FROM `{GBQ_PROJECT_ID}.{GBQ_DATASET_ID}.cur_pg_db_embeddings`)
        LIMIT {max_n_videos_to_update_embeddings}
    )
    GROUP BY
    embedding.id,
    embedding.video_url,
    embedding.embedding_type,
    embedding.start_segment,
    embedding.end_segment,
    embedding.segment_length,
    embedding.gcs_uri
    """

    # Execute the above query
    # TODO: UNCOMMENT THIS .head(1000) TO GET ALL OF THE EMBEDDINGS
    embeddings_df = read_gbq(embeddings_query)

    # Indicate that we're deleting the temporary GBQ tables
    logger.info("DELETING TEMPORARY GBQ TABLES...")

    # Delete each of the cur_pg_db tables
    try:
        read_gbq(
            f"DROP TABLE IF EXISTS `{GBQ_PROJECT_ID}.{GBQ_DATASET_ID}.cur_pg_db_video_metadata`"
        )
    except:
        pass

    try:
        read_gbq(
            f"DROP TABLE IF EXISTS `{GBQ_PROJECT_ID}.{GBQ_DATASET_ID}.cur_pg_db_transcriptions`"
        )
    except:
        pass

    try:
        read_gbq(
            f"DROP TABLE IF EXISTS `{GBQ_PROJECT_ID}.{GBQ_DATASET_ID}.cur_pg_db_embeddings`"
        )
    except:
        pass

    # ===============================
    # DOWNLOADING EMBEDDINGS FROM GCS
    # ===============================
    # Below, I'm going to download the embeddings from GCS to the local machine

    # Log that we're downloading the embeddings from GCS
    logger.info("DOWNLOADING EMBEDDINGS FROM GCS...")

    # Create a temporary directory to store the embeddings
    temp_emb_directory_path = Path("temp_embeddings")
    temp_emb_directory_path.mkdir(exist_ok=True, parents=True)

    # Remove any files that're already in the directory if it exists
    for file in temp_emb_directory_path.glob("*"):
        file.unlink()

    # Create a GCS client
    gcs_client = storage.Client(project=GBQ_PROJECT_ID)

    # Prepare the list of GCS URIs
    gcs_uris = embeddings_df["gcs_uri"].unique()

    def download_embedding(idx_and_uri):
        idx, gcs_uri = idx_and_uri
        try:
            # Parse the GCS URI
            split_gcs_uri = gcs_uri.split("gs://")[-1]
            bucket_name, file_name = split_gcs_uri.split("/")[0], "/".join(
                split_gcs_uri.split("/")[1:]
            )

            # Download the embedding corresponding with this GCS URI
            download_file_from_bucket(
                bucket_name=bucket_name,
                file_name=file_name,
                destination_folder=str(temp_emb_directory_path) + "/",
                project_id=GBQ_PROJECT_ID,
                gcs_client=gcs_client,
                logger=logger,
            )

        except Exception as e:
            print(f"Error parsing GCS URI: {e}")
            pass

    # Use ThreadPoolExecutor to parallelize the download process
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(download_embedding, idx_uri): idx_uri
            for idx_uri in enumerate(gcs_uris)
        }
        for future in tqdm(as_completed(futures), total=len(gcs_uris)):
            pass

    # ===========================
    # LOADING EMBEDDINGS INTO RAM
    # ===========================
    # Below, I'm going to load the embeddings into RAM, so that I can then upload them to the Postgres database

    def load_embeddings_into_postgres(emb_file_paths):
        """
        This is a helper function that'll load a set of embeddings
        into RAM, and then into the Postgres database. By splitting this
        process into its own function, I can ensure that there isn't a
        huge memory spike when loading in the embeddings.
        """

        # We'll store the embeddings in a dictionary, where the key is the
        # embedding ID and the value is the ndarray of the embedding
        embeddings = {}
        for emb_file in emb_file_paths:
            try:
                # Load in the .npy file as a numpy array
                embedding = np.load(emb_file)

                # If the embedding is empty, skip it
                if embedding.shape == ():
                    continue

                # Get the embedding ID
                embedding_id = emb_file.stem

                # Add the embedding to the dictionary, storing the list representation
                embeddings[embedding_id] = embedding.tolist()

            except Exception as e:
                logger.error(f"Error loading embedding: {e}")
                pass

        # Delete all of the files in the temp directory
        for emb_file in emb_file_paths:
            emb_file.unlink()

        # Make a "loaded_embeddings_df" that has the embeddings loaded in
        loaded_embeddings_df = embeddings_df.copy()
        loaded_embeddings_df["embedding"] = loaded_embeddings_df["id"].apply(
            lambda x: embeddings.get(x, None)
        )

        # Drop any rows where the embedding is None
        loaded_embeddings_df = loaded_embeddings_df.dropna(
            subset=["embedding"]
        ).drop_duplicates(subset=["id"])

        # Determine which rows of the embeddings_df we need to add to the database
        embeddings_df_to_add = loaded_embeddings_df[
            ~loaded_embeddings_df["id"].isin(cur_database_embeddings_df["id"])
        ].copy()

        # Upload the embeddings to the database
        upload_to_table(
            embeddings_df_to_add.drop(columns=["gcs_uri"]),
            "embeddings",
            engine=engine,
            logger=logger,
            chunksize=postgres_upload_chunksize,
        )

    # Log that we're loading the embeddings into the Postgres database
    logger.info("LOADING EMBEDDINGS INTO POSTGRES...")

    # Break up the embeddings into chunks
    all_emb_file_names = list(temp_emb_directory_path.iterdir())
    for i in range(0, len(all_emb_file_names), n_max_embeddings_per_upload):
        chunk = all_emb_file_names[i : i + n_max_embeddings_per_upload]
        logger.debug(f"Loading embeddings chunk {i} into Postgres...")
        load_embeddings_into_postgres(chunk)

    # Delete the temp directory
    temp_emb_directory_path.rmdir()

    # ==============================
    # ADDING ROWS TO POSTGRES TABLES
    # ==============================
    # Below, I'm going to add the rows to the Postgres tables

    # Log that we're adding rows to the Postgres tables
    logger.info("ADDING ROWS TO POSTGRES TABLES...")

    # Determine which rows of the video_metadata_df we need to add to the database
    video_metadata_df_to_add = video_metadata_df[
        ~video_metadata_df["id"].isin(cur_database_video_metadata_df["id"])
    ].copy()

    # Determine which rows of the transcriptions_df we need to add to the database
    transcriptions_df_to_add = transcriptions_df[
        ~transcriptions_df["url"].isin(cur_database_transcriptions_df["url"])
    ].copy()

    # Log some information about the number of rows we're adding
    logger.info(
        f"Adding {len(video_metadata_df_to_add)} rows to the video_metadata table."
    )
    logger.info(
        f"Adding {len(transcriptions_df_to_add)} rows to the transcriptions table."
    )

    # Upload the video metadata to the database
    upload_to_table(
        video_metadata_df_to_add,
        "video_metadata",
        engine=engine,
        logger=logger,
    )

    # Upload the transcriptions to the database
    upload_to_table(
        transcriptions_df_to_add,
        "transcriptions",
        engine=engine,
        logger=logger,
    )

    # =========================
    # EMBEDDINGS INDEX CREATION
    # =========================
    # Below, I'm going to create the index for the embeddings table

    query_postgres(
        f"SET max_parallel_maintenance_workers = {postgres_max_parallel_maintenance_workers}; -- plus leader",
        engine=engine,
        logger=logger,
    )
    query_postgres(
        f"SET maintenance_work_mem = '{postgres_maintenance_work_mem}';",
        engine=engine,
        logger=logger,
    )

    # If the user wants to re-create the embeddings index, then do so
    if recreate_embeddings_index_if_exists:
        # Log that we're re-creating the embeddings index
        logger.info("RE-CREATING THE EMBEDDINGS INDEX...")

        # Drop the index if it already exists
        query_postgres(
            "DROP INDEX IF EXISTS embeddings_embedding_idx;",
            engine=engine,
            logger=logger,
        )

        # Run the query that will create the IVFFlat index
        query_postgres(
            f"""CREATE INDEX ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = {embeddings_index_ivfflat_nlist});""",
            engine=engine,
            logger=logger,
        )

    else:
        # Try to create the index if it doesn't exist
        try:
            # Log that we're creating the embeddings index
            logger.info("CREATING THE EMBEDDINGS INDEX...")

            # Run the query that will create the IVFFlat index
            query_postgres(
                f"""CREATE INDEX ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = {embeddings_index_ivfflat_nlist});""",
                engine=engine,
                logger=logger,
            )
        except Exception as e:
            # Log the error if the index already exists
            logger.error(f"An error occurred while creating the embeddings index: {e}")

    # ===========================
    # CREATING EMBEDDINGS TO TEXT
    # ===========================
    # Below, I'm going to create a table that maps embeddings to text

    # First, delete the table if it already exists
    delete_table("embeddings_to_text", engine, logger)

    table_creation_query = f"""
    CREATE TABLE embeddings_to_text AS (
        WITH 
        embedding_to_text_flattened AS (
            SELECT
            embeddings.id,
            embeddings.url,
            embeddings.start_segment,
            embeddings.end_segment,
            transcriptions.text,
            transcriptions.segment_id,
            transcriptions.segment_start,
            transcriptions.segment_end
            FROM
            embeddings
            LEFT JOIN
            transcriptions
            ON
            transcriptions.segment_id >= embeddings.start_segment
            AND
            transcriptions.segment_id < embeddings.end_segment
            AND
            transcriptions.url = embeddings.url
            ORDER BY
            url DESC,
            segment_id ASC
        ),
        
        embedding_to_text AS (
            SELECT
                id,
                url,
                start_segment,
                end_segment,
                ARRAY_TO_STRING(ARRAY_AGG(emb.text ORDER BY emb.segment_id), '') AS text,
                MIN(emb.segment_start) AS segment_start,
                MAX(emb.segment_end) AS segment_end
            FROM
                embedding_to_text_flattened emb
            GROUP BY
                id,
                url,
                start_segment,
                end_segment
        )

        SELECT * FROM embedding_to_text
    )
    """

    # Execute the above query
    query_postgres(
        table_creation_query,
        engine=engine,
        logger=logger,
    )


# ====================
# RUNNING THE FUNCTION
# ====================
# Below, I'm going to run the function that I've defined above.

# Run the function
if __name__ == "__main__":
    # Log that we're starting the database update
    logger.info("STARTING DATABASE UPDATE...")

    update_database()

    # Log that we're done
    logger.info("DATABASE UPDATE COMPLETE.")

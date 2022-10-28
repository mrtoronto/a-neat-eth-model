import json
import logging
from google.cloud import storage
from config.local_settings import bucket_id, firestore_creds

SCOPES = "https://www.googleapis.com/auth/cloud-platform"

def _init_creds():
	creds_filename = '/tmp/creds.json'
	with open(creds_filename, 'w') as f:
		json.dump(firestore_creds, f)

_init_creds()

storage_client = storage.Client.from_service_account_json('/tmp/creds.json')


bucket = storage_client.bucket(bucket_id)

def upload_blob(
    source_file_name, 
    folder_name = None
):
    """
    Uploads a file to Google Cloud Storage
    """
    ### If user specifies a specific folder, append that
    if folder_name:
        blob_location = f'{folder_name}/{source_file_name}'
    else:
        blob_location = f'{source_file_name}'

    ### Upload the file
    blob = bucket.blob(blob_location)
    blob.upload_from_filename(source_file_name)
    logging.info(f"File {source_file_name} uploaded to {blob_location}.")

def download_blob(
    source_blob_name, 
    destination_file_name
):
    """
    Downloads a file from Google Cloud Storage
    """
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    logging.info(f"Blob {blob.name} downloaded to {destination_file_name}.")
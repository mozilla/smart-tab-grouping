import glob
import os
from google.cloud import storage
import pandas as pd


def download_bucket_to_file(bucket_name, blob_path, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(destination_file_name)


def download_bucket_to_csv(bucket_name, blob_path):
    dest = "temp_path.csv"
    download_bucket_to_file(bucket_name, blob_path, dest)
    return pd.read_csv(dest)


def upload_directory(dir_path: str, bucket_name: str, destination_path: str, depth=0):
    print(f"Searching {dir_path}")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    for local_file in glob.glob(f"{dir_path}/**"):
        base_name = os.path.basename(local_file)
        if os.path.isfile(local_file):
            remote_path = os.path.join(destination_path, base_name)
            blob = bucket.blob(remote_path)
            print(f"Uploading {local_file} to {remote_path}")
            blob.upload_from_filename(local_file)
        else:
            if depth > 0:
                upload_directory(f"{dir_path}/{base_name}", bucket_name, f"{destination_path}/{base_name}", depth - 1)
            else:
                print(f"Not uploading deeper folder {local_file}")

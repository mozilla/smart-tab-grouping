import json
import os
import sys

import google_crc32c
from dotenv import load_dotenv
from google.cloud import secretmanager

LOCAL_ENVIRONMENT = "local"
LOCAL_KUBE_ENVIRONMENT = "local kube"
DEV_ENVIRONMENT = "dev"
STAGE_ENVIRONMENT = "stage"
PROD_ENVIRONMENT = "prod"


def get_project_id():
    return "moz-fx-mozsoc-ml-nonprod"

def get_environment_name():
    environment = get_environment()
    if environment == PROD_ENVIRONMENT:
        return "PROD"
    elif environment == STAGE_ENVIRONMENT:
        return "STAG"
    elif environment == DEV_ENVIRONMENT:
        return "DEV"
    elif environment == LOCAL_ENVIRONMENT:
        if all([arg in sys.argv for arg in ['run', '--with', 'kubernetes']]):
            return "LOCAL_KUBE"
        else:
            return "LOCAL"


def is_kubernetes_environment():
    local_kube = all([arg in sys.argv for arg in ['run', '--with', 'kubernetes']])

    argo_workflows = all([arg in sys.argv for arg in['argo-workflows']])

    pod_namesace =  'METAFLOW_KUBERNETES_POD_NAMESPACE' in os.environ

    return local_kube or argo_workflows or pod_namesace



def get_environment():
    try:
        return os.environ["METAFLOW_KUBERNETES_POD_NAMESPACE"].split("-")[2]
    except:
        pass
    return LOCAL_ENVIRONMENT


def is_local_environment():
    return get_environment() == LOCAL_ENVIRONMENT


def is_remote_environment():
    return get_environment() != LOCAL_ENVIRONMENT


def load_secret(secret_id: str) -> str:
    client = secretmanager.SecretManagerServiceClient()
    secret_path = client.secret_version_path(get_project_id(), secret_id, "latest")
    response = client.access_secret_version(request={"name": secret_path})
    crc32c = google_crc32c.Checksum()
    crc32c.update(response.payload.data)
    if response.payload.data_crc32c != int(crc32c.hexdigest(), 16):
        raise Exception(f"Secret CRC Corrupted in project {get_project_id()} and path {secret_path}")
    return response.payload.data.decode("UTF-8")


def load_remote_env():
    # Load secrets
    json_secrets = ['metaflow-job-secrets']
    for secret_id in json_secrets:
        raw_env = load_secret(secret_id)
        envs = json.loads(raw_env)
        for k, v in envs.items():
            os.environ[k.upper()] = v

    print(f"Loaded secrets from {get_project_id()}")


def load_local_env():
    did_load = load_dotenv(override=True)
    if not did_load:
        if is_local_environment():
            print("Did not load from .env file or file was empty")
    else:
        print("Loading env from .env")


def load_env():
    """
    Load all from the remote env if available.  Then overwrite the local .env
    """
    load_remote_env()
    load_local_env()

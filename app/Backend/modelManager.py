# pylint: disable=W0105
# pylint: disable=W0012
# pylint: disable=unused-import
# pylint: disable=no-name-in-module

import os
from huggingface_hub import snapshot_download

base_cached_path = './cached_huggingface_models/models--google--flan-t5-base/refs/main'
xl_cached_path = './cached_huggingface_models/models--google--flan-t5-xl/refs/main'


def prepare_models():
    """
    Downloads and prepares 'flan-t5-base' and 'flan-t5-xl' models from Hugging Face model hub.
    The models are downloaded and stored in a cache directory.
    If the models have already been downloaded (checked via existing paths), this function won't re-download the models.

    Parameters:
    None

    Returns:
    None: This function does not return any value but downloads and prepares the required models in the specified cache directory.
    """

    snapshot_download(
        repo_id="google/flan-t5-base",
        cache_dir="./cached_huggingface_models",
        local_files_only=os.path.exists(base_cached_path)
    )

    snapshot_download(
        repo_id="google/flan-t5-xl",
        cache_dir="./cached_huggingface_models",
        local_files_only=os.path.exists(xl_cached_path)
    )

# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from urllib.parse import urlparse
import logging

import platformdirs
import requests

from config import CACHED_URLS

CACHE_PATH = Path(platformdirs.user_cache_path("discord-bot"))


def _get_file_name(url: str) -> Path:
    parsed = urlparse(url)
    file_path = Path(CACHE_PATH, parsed.path[1:])
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


def add_to_cache(url: str) -> bytes:
    file_path = _get_file_name(url)
    if file_path.is_file():
        return file_path.read_bytes()

    attachment = requests.get(url, allow_redirects=True)
    attachment.raise_for_status()

    with open(file_path, mode="wb") as file:
        file.write(attachment.content)
    return attachment.content


def get_from_cache(url: str) -> bytes:
    file_path = _get_file_name(url)
    if file_path.is_file():
        return file_path.read_bytes()
    elif not file_path.exists():
        return add_to_cache(url)
    raise FileNotFoundError("Name already used by not-file.")


def load_pre_cached_urls():
    """Ensures that all URLs in CACHED_URLS are in the cache."""
    logging.info("Caching files...")
    for url in CACHED_URLS:
        try:
            add_to_cache(url)
        except Exception:
            logging.exception("Coud not cache %s", url)
    logging.info("Caching complete.")


def get_pre_cached_content() -> str:
    """Returns the content of all pre-cached URLs as a single string."""
    content = []
    for url in CACHED_URLS:
        file_path = _get_file_name(url)
        if file_path.is_file():
            content.append(file_path.read_text())
    return "\n".join(content)

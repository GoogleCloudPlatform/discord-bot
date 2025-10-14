from pathlib import Path
from urllib.parse import urlparse

import platformdirs
import requests

CACHE_PATH = platformdirs.user_cache_path("discord-bot")


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

# Typical attachment uri: "https://cdn.discordapp.com/attachments/1196554310318837922/1288883797793837097/0_0_productGfx_e15bc0cfc61b36f94fd76f982d932f4b.jpg?ex=66fe0e53&is=66fcbcd3&hm=b2af6bee0655286b2acff5a98a6f2d2cd1f03b268d6f3f5b144053923a37b005&"
# Extract channel id, message id and filename, save the content as b64 or bytes
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

    with open(file_path, mode='wb') as file:
        file.write(attachment.content)
    return attachment.content


def get_from_cache(url: str) -> bytes:
    file_path = _get_file_name(url)
    if file_path.is_file():
        return file_path.read_bytes()
    elif not file_path.exists():
        return add_to_cache(url)
    raise FileNotFoundError("Name already used by not-file.")

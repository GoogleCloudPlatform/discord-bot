# Copyright 2024 Google LLC
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

import pathlib
import json

import platformdirs

# Collect user consent before allowing them to interact with the AI.
# They need to first send a Direct Message to the bot to confirm that they
# agree for their requests to be transferred to Google's AI.
_CONFIG_FOLDER = platformdirs.user_config_dir(
    'ExampleGeminiDiscordBot', 'GoogleCloud'
)
_USER_CONSENTS_FILE = pathlib.Path(_CONFIG_FOLDER, "user_consents.txt")
print(f"Loading user consents from {_USER_CONSENTS_FILE}...")
try:
    _USER_CONSENTS = set(json.loads(_USER_CONSENTS_FILE.read_text()))
    print(f"{len(_USER_CONSENTS)} consents loaded.")
except FileNotFoundError:
    print("The file was not found.")
    _USER_CONSENTS = set()


def save_user_consents(_consent_file: str| pathlib.Path = _USER_CONSENTS_FILE) -> None:
    """
    Save the list of users that consent to interact with the AI to _USER_CONSENTS_FILE
    """
    user_ids = list(_USER_CONSENTS)
    user_ids.sort()
    _USER_CONSENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_USER_CONSENTS_FILE, mode="w") as output_file:
        json.dump(user_ids, output_file)
    return


def check_user_consent(user_id: int) -> bool:
    """
    Checks if the user agreed to interact with the AI.
    """
    return user_id in _USER_CONSENTS


def record_user_consent(user_id: int) -> None:
    """
    Adds the user id to the list of users that have agreed to interact with the AI.
    """
    _USER_CONSENTS.add(user_id)
    save_user_consents()
    return

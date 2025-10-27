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

import logging
from pathlib import Path

import dotenv

logging.basicConfig(level=logging.INFO)

from agents import cache

logger = logging.getLogger(__name__)

# Load .env file from the same directory as main.py
dotenv_path = Path(__file__).parent / ".env"
dotenv.load_dotenv(dotenv_path=dotenv_path, verbose=True)

import discord_engine

if __name__ == "__main__":
    logger.info("Starting the Discord bot...")

    # Configure ADK
    # TODO any pre-flight stuff, verify config, etc.

    # Launch the discord_wrapped ADK Agent
    discord_engine.start_discord_engine()

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

"""This file contains the configuration for the ADK bot."""
#  GOOGLE_GENAI_USE_VERTEXAI=TRUE
#  GOOGLE_CLOUD_PROJECT="TODO"
#  GOOGLE_CLOUD_LOCATION="TODO"

# Application
APP_NAME = "DiscordBotADK"
USER = "public"

# Bot
BOT_NAME = "Agent Wiskers"
SLASH_COMMAND_STOP = "agent_stop"

# Models
DECISION_AGENT_MODEL = "gemini-2.5-flash-lite"
ROOT_AGENT_MODEL = "gemini-2.5-flash-lite"

# Agents
DECISION_AGENT_NAME = "the_great_decider"
DECISION_AGENT_DESCRIPTION = (
    "This agent is used to decide should the system answer a user message or not."
)
DECISION_AGENT_OUTPUT_KEY = "initial_decision"

ROOT_AGENT_NAME = "base_agent"
ROOT_AGENT_DESCRIPTION = "General purpose agent to generate some responses."

# Event Compaction
COMPACTION_INTERVAL = 2
OVERLAP_SIZE = 1

# URLs for Knowledge in Context
CACHED_URLS = [
    "https://raw.githubusercontent.com/GoogleCloudPlatform/agent-starter-pack/refs/heads/main/agent_starter_pack/resources/docs/adk-cheatsheet.md",
    "https://raw.githubusercontent.com/google/adk-python/refs/heads/main/CHANGELOG.md",
    "https://raw.githubusercontent.com/google/adk-python/refs/heads/main/llms-full.txt",
    "https://raw.githubusercontent.com/google/adk-python/main/contributing/samples/hello_world_app/main.py",
    "https://raw.githubusercontent.com/google/adk-python/main/contributing/samples/hello_world_app/agent.py",
]

# Sources of Knowledge for RAG/Search
# TODO https://github.com/google/adk-python/tree/main/contributing/samples/adk_knowledge_agent
VERTEXAI_DATASTORE_ID="TODO"
VERTEXAI_SEARCH_ENGINE_ID="TODO"
#  data_store_id: The Vertex AI search data store resource ID in the format of
#  "projects/{project}/locations/{location}/collections/{collection}/dataStores/{dataStore}".
#  search_engine_id: The Vertex AI search engine resource ID in the format of
#  "projects/{project}/locations/{location}/collections/{collection}/engines/{engine}".
RAG_TOOL_PATH = "TODO"
RAG_SOURCES = [
    "https://raw.githubusercontent.com/GoogleCloudPlatform/agent-starter-pack/refs/heads/main/agent_starter_pack/resources/docs/adk-cheatsheet.md",
    "https://raw.githubusercontent.com/google/adk-python/refs/heads/main/CHANGELOG.md",
    "https://raw.githubusercontent.com/google/adk-python/refs/heads/main/llms-full.txt",
    "https://raw.githubusercontent.com/google/adk-python/main/contributing/samples/hello_world_app/main.py",
    "https://raw.githubusercontent.com/google/adk-python/main/contributing/samples/hello_world_app/agent.py",
    # TODO consider a list of samples as cached context.
    # "https://github.com/google/adk-python/tree/main/contributing/samples",
    # TODO a list of all whitepaper content in Vertex AI Search datastore
]

# MIME Types
ACCEPTED_MIMES = {
    "application/pdf",
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "image/gif",
    "image/png",
    "image/jpeg",
    "image/webp",
    "text/plain",
    "video/mov",
    "video/mpeg",
    "video/mp4",
    "video/mpg",
    "video/avi",
    "video/wmv",
    "video/mpegps",
    "video/flv",
}

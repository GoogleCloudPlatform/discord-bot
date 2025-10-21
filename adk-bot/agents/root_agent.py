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

import hikari
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.tools import google_search
from google.genai.types import Content

from .common import _APP_NAME, session_service, ensure_session_exists, _USER, message_to_content, load_prompt

root_agent = Agent(
    name="base_agent",
    model="gemini-2.5-flash-lite",
    description=(
        "General purpose agent to generate some responses."
    ),
    instruction=(
        load_prompt('root_agent')
    ),
    tools=[google_search]
)

_runner = Runner(
    agent=root_agent,
    app_name=_APP_NAME,
    session_service=session_service,
)

async def reply_to_message(message: hikari.Message, channel_id: int) -> Content:
    await ensure_session_exists(channel_id)

    new_message = message_to_content(message)
    for event in _runner.run(user_id=_USER, session_id=str(channel_id), new_message=new_message):
        if event.is_final_response():
            return event.content
    raise RuntimeError
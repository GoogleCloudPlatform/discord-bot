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

import json

import hikari
import pydantic
from google.adk.agents import Agent
from google.adk.runners import Runner

from .common import _APP_NAME, session_service, ensure_session_exists, _USER, message_to_content, load_prompt


class DecisionOutput(pydantic.BaseModel):
    answer: bool
    thread_title: str

decision_agent = Agent(
    name="the_great_decider",
    model="gemini-2.5-flash-lite",
    description=(
        "This agent is used to decide should the system answer a user message or not."
    ),
    instruction=load_prompt('decision_agent_prompt'),
    output_schema=DecisionOutput,
    output_key="initial_decision"
)

_decision_runner = Runner(
    agent=decision_agent,
    app_name=_APP_NAME,
    session_service=session_service,
)

async def decide_on_answer(message: hikari.Message) -> DecisionOutput:
    await ensure_session_exists(message.id)

    new_message = message_to_content(message)

    for event in _decision_runner.run(user_id=_USER, session_id=str(message.id), new_message=new_message):
        if event.is_final_response():
            await session_service.delete_session(app_name=_APP_NAME, user_id=_USER, session_id=str(message.id))
            return DecisionOutput(**json.loads(event.content.parts[0].text))
    raise RuntimeError
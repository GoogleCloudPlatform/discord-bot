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

from .common import session_service, ensure_session_exists, message_to_content, load_prompt, USER, delete_session
from config import (
    DECISION_AGENT_NAME,
    DECISION_AGENT_MODEL,
    DECISION_AGENT_DESCRIPTION,
    DECISION_AGENT_OUTPUT_KEY, APP_NAME,
)

class DecisionOutput(pydantic.BaseModel):
    answer: bool
    thread_title: str

decision_agent = Agent(
    name=DECISION_AGENT_NAME,
    model=DECISION_AGENT_MODEL,
    description=DECISION_AGENT_DESCRIPTION,
    instruction=load_prompt('decision_agent_prompt'),
    output_schema=DecisionOutput,
    output_key=DECISION_AGENT_OUTPUT_KEY,
)

async def decide_on_answer(message: hikari.Message) -> DecisionOutput:
    """
    Based on the message received, decide if the bot should start a conversation.

    Returns:
        DecisionOutput object with indication if the bot should answer, and if that's True it will also contain
        thread_title set to a short title that matches the topic of the message.
    """
    from app import decision_app
    decision_runner = Runner(
        app=decision_app,
        session_service=session_service,
    )

    session = await ensure_session_exists(message.id)
    new_message = message_to_content(message)

    for event in decision_runner.run(user_id=session.user_id, session_id=session.id, new_message=new_message):
        if event.is_final_response():
            await delete_session(session.id)
            return DecisionOutput(**json.loads(event.content.parts[0].text))
    raise RuntimeError
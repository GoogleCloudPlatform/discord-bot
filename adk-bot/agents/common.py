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
import logging

import hikari
from google.adk.events import Event
from google.adk.sessions import InMemorySessionService, Session
from google.genai.types import Content, Part

from . import cache

_APP_NAME = "DiscordBot"
_USER = "public"

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

session_service = InMemorySessionService()

_SHARED_PROMPT = open("agents/prompts/shared_prompt.txt").read()

def load_prompt(prompt_name: str) -> str:
    """
    Loads the prompt from a file with given name, prepending it with the shared prompt that describes the whole system
    to all the agents.
    """
    return _SHARED_PROMPT + open(f"agents/prompts/{prompt_name}.txt").read()

def message_to_parts(message: hikari.Message) -> list[Part]:
    author = getattr(message.member, 'display_name', None) or message.author.username
    if message.author.is_bot:
        parts = [Part.from_text(text=message.content)]
    else:
        parts = [Part.from_text(text=json.dumps({"author": author, "content": message.content}))]

    for attachment in message.attachments:
        if attachment.media_type not in ACCEPTED_MIMES:
            part = Part.from_text(
                text=f"Here user uploaded a file in unsupported {attachment.media_type} type."
            )
        else:
            data = cache.get_from_cache(attachment.url)
            part = Part.from_bytes(data=data, mime_type=attachment.media_type)
        parts.append(part)

    return parts

def message_to_content(message: hikari.Message, role: str = "user") -> Content:
    parts = message_to_parts(message)

    return Content(role=role, parts=parts)

async def load_session_with_messages(channel_id: int, messages: list[hikari.Message]):
    contents = [Content(role="user", parts=[])]

    session = await ensure_session_exists(channel_id)
    for message in reversed(messages):
        if message.author.is_bot and contents[-1].role == "user":
            contents.append(Content(role="model", parts=[]))
        elif (not message.author.is_bot) and contents[-1].role == "model":
            contents.append(Content(role="user", parts=[]))
        contents[-1].parts.extend(message_to_parts(message))
    for content in contents:
        await session_service.append_event(session=session, event=Event(content=content, author=content.role))


async def ensure_session_exists(channel_id: int) -> Session:
    if (session := await session_service.get_session(app_name=_APP_NAME, user_id=_USER, session_id=str(channel_id))) is None:
        logging.info(f"Creating session for channel {channel_id}.")
        session = await session_service.create_session(app_name=_APP_NAME, user_id=_USER, session_id=str(channel_id), state={'bot_name': 'Agent Whiskers'})
    return session

async def session_exists(channel_id: int) -> bool:
    result = await session_service.get_session(app_name=_APP_NAME, user_id=_USER, session_id=str(channel_id))
    return result is not None
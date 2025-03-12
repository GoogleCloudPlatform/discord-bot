import json
import logging
import os
import time
from collections import defaultdict, deque
from typing import Literal

import google.auth
import hikari
from google import genai
from google.genai.types import (
    Part,
    Content,
    GenerateContentConfig,
    GenerateContentResponse,
    Tool,
    GoogleSearch,
)
from hikari import OwnUser

import discord_cache

VERTEX_TOS = "https://developers.google.com/terms"
GEMINI_TOS = "https://ai.google.dev/gemini-api/terms"

GEMINI_MODEL_NAME = "gemini-2.0-flash-exp"

_MAX_HISTORY_TOKEN_SIZE = 1000000  # 1M to keep things simple, real limit is 1,048,576

_LOGGER = logging.getLogger("bot.gemini")

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

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)

if GEMINI_API_KEY is None:
    _client = genai.Client(
        vertexai=True, project=google.auth.default()[1], location="us-central1"
    )
else:
    _client = genai.Client(api_key=GEMINI_API_KEY)

_gen_config = GenerateContentConfig(
    temperature=0,
    candidate_count=1,
    max_output_tokens=1800,
    system_instruction=f"You are a Discord bot named GeminiBot."
    "Your task is to provide useful information to users interacting with you. "
    "You should be positive, cheerful and polite. "
    "Feel free to use the default Discord emojis. "
    "You are provided with chat history in JSON format, but your answers should be regular text. "
    "Always reply to the last message in the chat history. ",
    tools=[Tool(google_search=GoogleSearch())],  # for Gemini 2
    response_modalities=["TEXT"],
)


class ChatPart:
    """
    ChatPart is used to internally store the history of communication in various Discord channels.
    We save the chat message as `vertexai.generative_models.Part` object (ready to use in communication with Gemini),
    `role` tells us if given message was made by our bot ("model") or a user ("user"). The object also stores the token
    count, so we don't need to re-calculate it.

    Currently, our code handles only text interactions, however the Part objects can represent images, videos and audio
    files, which will be useful in the future.
    """

    def __init__(
        self, chat_part: Part, role: Literal["user", "model"], token_count: int
    ):
        self.part = chat_part
        self.role = role
        self.token_count = token_count or 0

    def __str__(self):
        return f"<{self.role}: {self.part} [{self.token_count}]>"

    def __repr__(self):
        return str(self)

    @staticmethod
    def _count_tokens(part: Part) -> int:
        if hasattr(part, "text") and part.text is not None:
            return int(len(part.text) * 0.3)
        if hasattr(part, "inline_data") and part.inline_data is not None:
            if part.inline_data.mime_type.startswith("image"):
                return 258
            elif part.inline_data.mime_type.startswith("video"):
                return int(len(part.inline_data.data) / 1000)
            elif part.inline_data.mime_type.startswith("audio"):
                return int(len(part.inline_data.data) / 800)
        # 1000 bytes per token video, 800 bytes per token for audio - those are rough estimates
        _LOGGER.debug(f"Counting tokens for {part}"[:200])
        start = time.time()
        count = _client.models.count_tokens(
            model=GEMINI_MODEL_NAME, contents=part
        ).total_tokens
        _LOGGER.debug(
            f"Counted tokens for {part} in {time.time() - start}s with token count: {count}"
        )
        return count

    @classmethod
    def from_user_chat_message(cls, message: hikari.Message) -> list["ChatPart"]:
        """
        Create a user ChatPart object from hikari.Message.

        Stores the text content of the message as JSON encoded object and assigns the `role` as "user".
        This method also calculates and saves the token count.
        """
        msg = json.dumps(
            {
                "author": getattr(message.member, "display_name", False)
                or message.author.username,
                "content": message.content,
            }
        )
        text_part = Part.from_text(text=msg)
        parts = [(text_part, cls._count_tokens(text_part))]

        for a in message.attachments:
            if a.media_type not in ACCEPTED_MIMES:
                part = Part.from_text(
                    text=f"Here user uploaded a file in unsupported {a.media_type} type."
                )
            else:
                data = discord_cache.get_from_cache(a.url)
                part = Part.from_bytes(data=data, mime_type=a.media_type)
            parts.append((part, cls._count_tokens(part)))

        return [cls(part, "user", tokens) for part, tokens in parts]

    @classmethod
    def from_bot_chat_message(cls, message: hikari.Message) -> list["ChatPart"]:
        """
        Create a model ChatPart object from hikari.Message.

        Stores the text content of the message and assigns the `role` as "model".
        This method also calculates and saves the token count.
        """
        part = Part.from_text(text=message.content)
        tokens = cls._count_tokens(part)

        parts = [(part, tokens)]

        for a in message.attachments:
            data = discord_cache.get_from_cache(a.url)
            part = Part.from_bytes(data=data, mime_type=a.media_type)
            parts.append((part, cls._count_tokens(part)))

        return [cls(part, "model", tokens) for part, tokens in parts]

    @classmethod
    def from_ai_reply(cls, response: GenerateContentResponse | Part) -> "ChatPart":
        """
        Create a model ChatPart object from Gemini response.

        Stores the text content of the message and assigns the `role` as "model".
        Saves the token count from the model response.
        """
        part = Part.from_text(text=response.text)
        if isinstance(response, GenerateContentResponse):
            tokens = response.usage_metadata.candidates_token_count
        else:
            tokens = cls._count_tokens(part)
        return cls(part, "model", tokens)

    @classmethod
    def from_raw_part(
        cls, part: Part, role: Literal["user", "model"] = "model"
    ) -> "ChatPart":
        """
        Create a model ChatPart object from a raw Part.

        Stores the whole part and assigns the `role` as "model".
        Saves the token count using model call.
        """
        return cls(part, role, cls._count_tokens(part))

    @classmethod
    def from_bytes(cls, data: bytes, mime_type: str) -> "ChatPart":
        """
        Create a model ChatPart object to represent attached image.

        Stores the bytes content and assigns the `role` as "model".
        Saves the token count by querying Gemini model.
        """
        part = Part.from_bytes(bytes=data, mime_type=mime_type)
        return cls(part, "model", cls._count_tokens(part))


class ChatHistory:
    """
    Object of this class keeps track of the chat history in a single Discord channel by
    storing a deque of ChatPart objects.
    """

    def __init__(self):
        self._history: deque[ChatPart] = deque()

    async def add_message(self, message: hikari.Message) -> None:
        """
        Create a new ChatPart object and append it to the chat history.
        """
        self._history.extend(ChatPart.from_user_chat_message(message))

    async def load_history(self, channel: hikari.GuildTextChannel, bot_id: int) -> None:
        """
        Reads chat history of a given channel and stores it internally.

        The history will be read until the history exceeds the token limit for the Gemini model.

        :param channel: Guild channel that will be read.
        :param bot_id: The ID of the bot that we are running as. Needed to properly recognize responses from
            previous sessions.
        """
        _LOGGER.info(f"Loading history for: {channel} {type(channel)}")
        guild = channel.get_guild()
        member_cache = {}
        tokens = 0
        messages = 0
        async for message in channel.fetch_history():
            messages += 1
            if messages > 50:
                # To speed up the starting process, just read only the last 50 messages
                break
            if message.author.id not in member_cache:
                member_cache[message.author.id] = guild.get_member(message.author.id)
            message.member = member_cache[message.author.id]
            if message.author.id == bot_id:
                self._history.extendleft(
                    reversed(ChatPart.from_bot_chat_message(message))
                )
            else:
                self._history.extendleft(
                    reversed(ChatPart.from_user_chat_message(message))
                )
            tokens += self._history[0].token_count or 0
            if tokens > _MAX_HISTORY_TOKEN_SIZE:
                break
        _LOGGER.info(f"History loaded for {channel}.")

    def _build_content(self) -> (list[Content], int):
        """
        Prepare the whole Content structure to be sent to the AI model, containing the whole chat history so far
        (or as much as the token limit allows).

        The Gemini model accepts a sequence of Content objects, each Content object contains one or more Part objects.
        Content objects have the `role` attribute that tells Gemini who's the author of a given piece of conversation
        history. The model expects that the sequence of incoming Content objects is a conversation between "model" and
        "user" - in our case, we combine all user messages into single Content object, with proper attribution, so that
        Gemini can recognize who said what. Model Content objects are sent as regular text.
        """
        # Buffer keeps tuples of (part, role)
        buffer = deque()
        contents = deque()
        tokens = 0
        parts_count = 0

        for part in reversed(self._history):
            parts_count += 1
            if buffer and buffer[0][1] != part.role:
                content = Content(role=buffer[0][1], parts=list(b[0] for b in buffer))
                contents.appendleft(content)
                buffer.clear()
            buffer.appendleft((part.part, part.role))
            tokens += part.token_count or 0

            if tokens > _MAX_HISTORY_TOKEN_SIZE:
                _LOGGER.info("Memory full, will purge now.")
                break

        # We fit whole _history in the contents, no need to clear memory
        if buffer:
            user_content = Content(role=buffer[0][1], parts=list(b[0] for b in buffer))
            contents.appendleft(user_content)

        # We need to forget the tail of history, so we don't waste memory
        for _ in range(len(self._history) - parts_count):
            self._history.popleft()

        while contents[0].role == "model" or len(contents[0].parts) == 0:
            # Can't have model start the conversation
            contents.popleft()

        return list(contents), tokens

    async def trigger_answer(self) -> (list[str], list[hikari.Bytes]):
        """
        Uses AI to generate answer to the current chat history. Will handle function calling if the model
        requests functions to be called.

        Note: The last message in the chat history has to be from a user.
        """
        if self._history[-1].role == "model":
            raise RuntimeError(
                "Last message in chat history needs to be from a user to generate a reply."
            )

        content, tokens = self._build_content()

        _LOGGER.info(f"Generating answer for estimated {tokens} tokens...")
        start = time.time()

        response = await _client.aio.models.generate_content(
            model=GEMINI_MODEL_NAME, contents=content, config=_gen_config
        )

        _LOGGER.info(
            f"Generated response for estimated {tokens} tokens in {time.time()-start}s."
        )
        self._history.append(ChatPart.from_ai_reply(response))

        return [response.text], []


class GeminiBot:
    """
    Class representing the state of current instance of our Bot.

    It keeps track of its own identity, Gemini model configuration and chat history for all the channels it
    interacted with.
    """

    def __init__(self, me: OwnUser):
        self.me = me
        self.memory = defaultdict(ChatHistory)

    async def handle_message(self, event: hikari.GuildMessageCreateEvent) -> None:
        """
        Handle an incoming message. This method will save the message in bot's chat history and generate a reply if
        one is needed (bot was mentioned in the message).
        """
        # Do not respond to bots nor webhooks pinging us, only user accounts
        if event.author_id == self.me.id:
            return

        message = event.message

        if message.channel_id not in self.memory:
            chat_history = ChatHistory()
            await chat_history.load_history(await message.fetch_channel(), self.me.id)
            self.memory[message.channel_id] = chat_history
        else:
            # Loading history would catch this message anyway
            await self.memory[message.channel_id].add_message(message)

        if self.me.id not in event.message.user_mentions_ids:
            # Reply only when mentioned in the message.
            return

        # The bot has been pinged, we need to reply
        await event.get_channel().trigger_typing()
        try:
            text_responses, attachments = await self.memory[
                message.channel_id
            ].trigger_answer()
        except Exception as e:
            await event.message.respond(
                "Sorry, there was an error processing an answer for you :("
            )
            raise e
        for text_response in text_responses[:-1]:
            await event.message.respond(text_response[:2000])
        await event.message.respond(text_responses[-1][:2000], attachments=attachments)

import dataclasses
import json
import sys
import tempfile
from collections import defaultdict, deque
from typing import Literal, Any
import datetime
import pytz
import hikari
from hikari import OwnUser
from vertexai.generative_models import (
    GenerationResponse,
    Part,
    Content,
    GenerativeModel,
    GenerationConfig, Tool, FunctionDeclaration, ToolConfig,
)
from vertexai.preview import tokenization
from vertexai.vision_models import GeneratedImage

from imagen import generate_image_tool, call_generate_image

import discord_cache

VERTEX_TOS = "https://developers.google.com/terms"
GEMINI_MODEL_NAME = "gemini-1.5-pro-002"
_MAX_HISTORY_TOKEN_SIZE = (
    1000000  # 1M to keep things simple, real limit for Flash is 1,048,576
)

ACCEPTED_MIMES = {
    "application/pdf",
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
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

try:
    _tokenizer = tokenization.get_tokenizer_for_model(GEMINI_MODEL_NAME)
except ValueError:
    # _tokenizer needs to have the count_tokens() method
    _tokenizer = GenerativeModel(GEMINI_MODEL_NAME)




def get_current_time(timezone: str) -> str:
    """
    Get the current time in a specified timezone.

    Args:
        timezone: The timezone for which you want to check the time. For example: Europe/Warsaw.

    Returns:
        Current time in a given timezone, formatted in the YYYY-MM-DD HH:MM:SS format.
    """
    tz = pytz.timezone(timezone)
    tz_now = datetime.datetime.now(tz)
    return tz_now.strftime("%Y-%m-%d %H:%M:%S")

def call_get_current_time(call_part: Part) -> (str, None):
    assert call_part.function_call.name == "get_current_time"
    return get_current_time(call_part.function_call.args["timezone"]), None

def noop() -> str:
    """
    Do nothing, just allows you to normally reply to users, but in the function calling mode.

    :return:
        Nothing of value, not important.
    """
    return "Nothing happens"

def call_noop(call_part: Part) -> (str, None):
    return noop(), None

TOOL_CALLING = {
    'get_current_time': call_get_current_time,
    'generate_image': call_generate_image,
    'noop': call_noop,
}

TOOLS = [
    Tool([generate_image_tool, FunctionDeclaration.from_func(get_current_time), FunctionDeclaration.from_func(noop)]),
]

class ChatPart:
    """
    ChatPart is used to internally store the history of communication in various Discord channels.
    We save the chat message as `vertexai.generative_models.Part` object (ready to use in communication with Gemini),
    `role` tells us if given message was made by our bot ("model") or a user ("user"). The object also stores the token
    count, so we don't need to re-calculate it.

    Currently, our code handles only text interactions, however the Part objects can represent images, videos and audio
    files, which will be useful in the future.
    """
    _model = GenerativeModel(GEMINI_MODEL_NAME)

    def __init__(
        self, chat_part: Part, role: Literal["user", "model"], token_count: int
    ):
        self.part = chat_part
        self.role = role
        self.token_count = token_count

    def __str__(self):
        return f"<{self.role}: {self.part} [{self.token_count}]>"

    def __repr__(self):
        return str(self)

    @classmethod
    def _count_tokens(cls, part: Part) -> int:
        if hasattr(part, 'text'):
            return _tokenizer.count_tokens(part).total_tokens
        # Non-text parts need to call the API to count tokens
        return cls._model.count_tokens(part).total_tokens

    @classmethod
    def from_user_chat_message(cls, message: hikari.Message) -> list["ChatPart"]:
        """
        Create a user ChatPart object from hikari.Message.

        Stores the text content of the message as JSON encoded object and assigns the `role` as "user".
        This method also calculates and saves the token count.
        """
        msg = json.dumps(
            {"author": getattr(message.member, 'display_name', False) or message.author.username, "content": message.content}
        )
        text_part = Part.from_text(msg)
        parts = [(text_part, _tokenizer.count_tokens(text_part).total_tokens)]

        for a in message.attachments:
            if a.media_type not in ACCEPTED_MIMES:
                part = Part.from_text(f"Here user uploaded a file in unsupported {a.media_type} type.")
            else:
                data = discord_cache.get_from_cache(a.url)
                part = Part.from_data(data, a.media_type)
            parts.append((part, cls._count_tokens(part)))

        return [cls(part, "user", tokens) for part, tokens in parts]

    @classmethod
    def from_bot_chat_message(cls, message: hikari.Message) -> list["ChatPart"]:
        """
        Create a model ChatPart object from hikari.Message.

        Stores the text content of the message and assigns the `role` as "model".
        This method also calculates and saves the token count.
        """
        part = Part.from_text(message.content)
        tokens = _tokenizer.count_tokens(part).total_tokens

        parts = [(part, tokens)]

        for a in message.attachments:
            data = discord_cache.get_from_cache(a.url)
            part = Part.from_data(data, a.media_type)
            parts.append((part, cls._count_tokens(part)))

        return [cls(part, "model", tokens) for part, tokens in parts]

    @classmethod
    def from_ai_reply(cls, response: GenerationResponse | Part) -> "ChatPart":
        """
        Create a model ChatPart object from Gemini response.

        Stores the text content of the message and assigns the `role` as "model".
        Saves the token count from the model response.
        """
        part = Part.from_text(response.text)
        if isinstance(response, GenerationResponse):
            tokens = response.usage_metadata.candidates_token_count
        else:
            tokens = _tokenizer.count_tokens(part).total_tokens
        return cls(part, "model", tokens)

    @classmethod
    def from_raw_part(cls, part: Part) -> "ChatPart":
        """
        Create a model ChatPart object from a raw Part.

        Stores the whole part and assigns the `role` as "model".
        Saves the token count using model call.
        """
        return cls(part, "model", cls._count_tokens(part))

    @classmethod
    def from_bytes(cls, data: bytes, mime_type: str) -> "ChatPart":
        """
        Create a model ChatPart object to represent attached image.

        Stores the bytes content and assigns the `role` as "model".
        Saves the token count by querying Gemini model.
        """
        part = Part.from_data(data, mime_type)
        return cls(part, "model", cls._count_tokens(part))


class ChatHistory:
    """
    Object of this class keep track of the chat history in a single Discord channel by
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
        print("Loading history for: ", channel, type(channel))
        guild = channel.get_guild()
        member_cache = {}
        tokens = 0
        messages = 0
        async for message in channel.fetch_history():
            messages += 1
            if messages > 100:
                # To speed up the starting process, just read only the last 100 messages
                break
            if message.author.id not in member_cache:
                member_cache[message.author.id] = guild.get_member(message.author.id)
            message.member = member_cache[message.author.id]
            if message.author.id == bot_id:
                self._history.extendleft(
                    reversed(ChatPart.from_bot_chat_message(message))
                )
            else:
                self._history.extendleft(reversed(ChatPart.from_user_chat_message(message)))
            tokens += self._history[0].token_count
            if tokens > _MAX_HISTORY_TOKEN_SIZE:
                break

    def _build_content(self) -> list[Content]:
        """
        Prepare the whole Content structure to be sent to the AI model, containing the whole chat history so far
        (or as much as the token limit allows).

        The Gemini model accepts a sequence of Content objects, each Content object contains one or more Part objects.
        Content objects have the `role` attribute that tells Gemini who's the author of a given piece of conversation
        history. The model expects that the sequence of incoming Content objects is a conversation between "model" and
        "user" - in our case, we combine all user messages into single Content object, with proper attribution, so that
        Gemini can recognize who said what. Model Content objects are sent as regular text.
        """
        buffer = deque()
        contents = deque()
        tokens = 0
        parts_count = 0

        for part in reversed(self._history):
            parts_count += 1
            if part.role == "user":
                buffer.appendleft(part.part)
                tokens += part.token_count
            elif part.role == "model":
                if buffer:
                    user_content = Content(role="user", parts=list(buffer))
                    contents.appendleft(user_content)
                    buffer.clear()
                model_content = Content(role="model", parts=[part.part])
                contents.appendleft(model_content)
                tokens += part.token_count

            if tokens > _MAX_HISTORY_TOKEN_SIZE:
                print("Memory full, will purge now.")
                break

        # We fit whole _history in the contents, no need to clear memory
        if buffer:
            user_content = Content(role="user", parts=list(buffer))
            contents.appendleft(user_content)
            return list(contents)

        # We need to forget the tail of history, so we don't waste memory
        for _ in range(len(self._history) - parts_count):
            self._history.popleft()

        while contents[0].role == "model" or len(contents[0].parts) == 0:
            # Can't have model start the conversation
            contents.popleft()

        return list(contents)

    async def trigger_answer(self, model: GenerativeModel, force_tool_use: bool=False) -> (list[str], list[hikari.Bytes]):
        """
        Uses AI to generate answer to the current chat history.

        Note: The last message in the chat history has to be from a user.
        """
        if self._history[-1].role == "model":
            raise RuntimeError(
                "Last message in chat history needs to be from a user to generate a reply."
            )

        content = self._build_content()

        # for c in content:
        #     print("Role: ", c.role)
        #     for p in c.parts:
        #         print("  Part: ", str(p)[:100])

        # print(TOOLS)
        if force_tool_use:
            response = await model.generate_content_async(content, tools=TOOLS, tool_config=ToolConfig(
                function_calling_config=ToolConfig.FunctionCallingConfig(mode=ToolConfig.FunctionCallingConfig.Mode.ANY)))
        else:
            response = await model.generate_content_async(content, tools=TOOLS)

        print(response)

        discord_text_response = []
        discord_attachments = []
        call_results_parts = []
        call_requests_parts = []

        while any(part.function_call for part in response.candidates[0].content.parts):
            # Check if Gemini wants to call one of its TOOLS. With parallel function execution, there can be multiple parts
            # calling functions.
            for part in response.candidates[0].content.parts:
                if getattr(part, 'text', None):
                    self._history.append(ChatPart.from_ai_reply(part))
                    discord_text_response.append(part.text)

            for part in response.candidates[0].content.parts:
                if getattr(part, 'function_call', None):
                    print(part)
                    # self._history.append(ChatPart.from_raw_part(part))
                    call_requests_parts.append(part)
                    result, attachment = TOOL_CALLING[part.function_call.name](part)
                    if attachment:
                        discord_attachments.append(attachment)
                    response_part = Part.from_function_response(name=part.function_call.name, response={'content': result})
                    # self._history.append(ChatPart.from_raw_part(response_part))
                    call_results_parts.append(response_part)

            assert len(call_results_parts) > 0
            content.append(Content(parts=call_requests_parts, role="model")) # The Gemini request for function calls, all of them
            content.append(Content(parts=call_results_parts))
            # for c in content[-3:]:
            #     print("Role: ", c.role)
            #     for p in c.parts:
            #         print("  Part: ", str(p)[:200])
            # Sending only the original message, request for function call and function call response
            # To allow for very long function call responses
            response = await model.generate_content_async(content[-3:], tools=TOOLS)
            print("Responseee:", response)

        self._history.append(ChatPart.from_ai_reply(response))
        discord_text_response.append(response.text)

        discord_byte_attachments = []
        for attachment in discord_attachments:
            assert isinstance(attachment, GeneratedImage)
            with tempfile.NamedTemporaryFile(suffix=".png") as temp:
                attachment.save(temp.name)
                with open(temp.name, "rb") as image:
                    bytes = image.read()
            discord_byte_attachments.append(hikari.Bytes(bytes, "attachment.png", "image/png"))
            self._history.append(ChatPart.from_bytes(bytes, attachment._mime_type))


        return discord_text_response, discord_byte_attachments


class GeminiBot:
    """
    Class representing the state of current instance of our Bot.

    It keeps track of its own identity, Gemini model configuration and chat history for all the channels it
    interacted with.
    """
    def __init__(self, me: OwnUser):
        self.me = me
        self.model = GenerativeModel(
            model_name=GEMINI_MODEL_NAME,
            generation_config=GenerationConfig(temperature=0, max_output_tokens=1800),
            system_instruction=f"You are a Discord bot named GeminiBot or <@{self.me.id}>."
            "Your task is to provide useful information to users interacting with you. "
            "You should be positive, cheerful and polite. "
            "Feel free to use the default Discord emojis. "
            "You are provided with chat history in JSON format, but your answers should be regular text. "
            "Always reply to the last message in the chat history. "
            "Use the tools available to you to fulfill user requests. Don't hesitate to make the function calls!",
            tools=TOOLS,
        )
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

        # Check if the users wants to force usage of tools
        force_tools_use =  message.content.startswith("!")

        # The bot has been pinged, we need to reply
        await event.get_channel().trigger_typing()
        try:
            text_responses, attachments =  await self.memory[message.channel_id].trigger_answer(self.model, force_tools_use)
        except Exception as e:
            await event.message.respond("Sorry, there was an error processing your request :(")
            raise e
        for text_response in text_responses[:-1]:
            await event.message.respond(text_response[:2000])
        await event.message.respond(text_responses[-1][:2000], attachments=attachments)


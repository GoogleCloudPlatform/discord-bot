import json
import sys
from collections import defaultdict, deque
from typing import Literal

import hikari
from hikari import OwnUser
from vertexai.generative_models import (
    GenerationResponse,
    Part,
    Content,
    GenerativeModel,
    GenerationConfig,
)
from vertexai.preview import tokenization

VERTEX_TOS = "https://developers.google.com/terms"
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest"
_MAX_HISTORY_TOKEN_SIZE = (
    1000000  # 1M to keep things simple, real limit for Flash is 1,048,576
)

_tokenizer = tokenization.get_tokenizer_for_model(GEMINI_MODEL_NAME)


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
        self.token_count = token_count

    def __str__(self):
        return f"<{self.role}: {self.part} [{self.token_count}]>"

    def __repr__(self):
        return str(self)

    @classmethod
    def from_user_chat_message(cls, message: hikari.Message) -> "ChatPart":
        """
        Create a user ChatPart object from hikari.Message.

        Stores the text content of the message as JSON encoded object and assigns the `role` as "user".
        This method also calculates and saves the token count.
        """
        msg = json.dumps(
            {"author": message.member.display_name, "content": message.content}
        )
        part = Part.from_text(msg)
        tokens = _tokenizer.count_tokens(part).total_tokens
        return cls(part, "user", tokens)

    @classmethod
    def from_bot_chat_message(cls, message: hikari.Message) -> "ChatPart":
        """
        Create a model ChatPart object from hikari.Message.

        Stores the text content of the message and assigns the `role` as "model".
        This method also calculates and saves the token count.
        """
        part = Part.from_text(message.content)
        tokens = _tokenizer.count_tokens(part).total_tokens
        return cls(part, "model", tokens)

    @classmethod
    def from_ai_reply(cls, response: GenerationResponse) -> "ChatPart":
        """
        Create a model ChatPart object from Gemini response.

        Stores the text content of the message and assigns the `role` as "model".
        Saves the token count from the model response.
        """
        part = Part.from_text(response.text)
        tokens = response.usage_metadata.candidates_token_count
        return cls(part, "model", tokens)


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
        self._history.append(ChatPart.from_user_chat_message(message))

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
        async for message in channel.fetch_history():
            if message.author.id not in member_cache:
                member_cache[message.author.id] = guild.get_member(message.author.id)
            message.member = member_cache[message.author.id]
            if message.author.id == bot_id:
                self._history.appendleft(
                    ChatPart.from_bot_chat_message(message)
                )
            else:
                self._history.appendleft(ChatPart.from_user_chat_message(message))
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
                user_content = Content(role="user", parts=list(buffer))
                contents.appendleft(user_content)
                buffer.clear()
                model_content = Content(role="model", parts=[part.part])
                contents.appendleft(model_content)
                tokens += part.token_count

            if tokens > _MAX_HISTORY_TOKEN_SIZE:
                print("Memory full, will purge now.")
                break
        else:
            # We fit whole _history in the contents, no need to clear memory
            user_content = Content(role="user", parts=list(buffer))
            contents.appendleft(user_content)
            return list(contents)

        # We need to forget the tail of history, so we don't waste memory
        for _ in range(len(self._history) - parts_count):
            self._history.popleft()

        if contents[0].role == "model":
            # Can't have model start the conversation
            contents.popleft()

        return list(contents)

    async def trigger_answer(self, model: GenerativeModel) -> str:
        """
        Uses AI to generate answer to the current chat history.

        Note: The last message in the chat history has to be from a user.
        """
        if self._history[-1].role == "model":
            raise RuntimeError(
                "Last message in chat history needs to be from a user to generate a reply."
            )

        content = self._build_content()

        response = await model.generate_content_async(content)

        self._history.append(ChatPart.from_ai_reply(response))

        return response.text


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
            generation_config=GenerationConfig(temperature=1, max_output_tokens=1800),
            system_instruction=f"You are a Discord bot named GeminiBot or <@{self.me.id}>."
            "Your task is to provide useful information to users interacting with you. "
            "You should be positive, cheerful and polite. "
            "Feel free to use the default Discord emojis. "
            "You are provided with chat history in JSON format, but your answers should be regular text. "
            "Always reply to the last message in the chat history.",
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

        # The bot has been pinged, we need to reply
        await event.get_channel().trigger_typing()
        result = await self.memory[message.channel_id].trigger_answer(self.model)
        await event.message.respond(result[:2000])

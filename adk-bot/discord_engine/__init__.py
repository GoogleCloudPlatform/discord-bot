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
import logging.handlers
import os

import hikari
from hikari import ChannelType

from agents import reply_to_message, decide_on_answer, DecisionOutput, session_exists, load_session_with_messages
from . import memory
from config import BOT_NAME, SLASH_COMMAND_STOP

assert(os.getenv("DISCORD_BOT_TOKEN") is not None)

bot = hikari.GatewayBot(
    token=os.getenv("DISCORD_BOT_TOKEN"),
    intents=hikari.Intents.ALL_GUILDS_UNPRIVILEGED | hikari.Intents.MESSAGE_CONTENT,
)


@bot.listen()
async def on_ready(event: hikari.StartedEvent):
    """

    """
    application = await bot.rest.fetch_application()
    commands = [
        bot.rest.slash_command_builder(SLASH_COMMAND_STOP, f"Makes {BOT_NAME} stop participating in a thread."),
    ]

    await bot.rest.set_application_commands(application=application.id, commands=commands)

    logging.info("Discord bot is ready!")

@bot.listen()
async def handle_interactions(event: hikari.CommandInteractionCreateEvent) -> None:
    """Listen for slash commands being executed."""
    if event.interaction.command_name == SLASH_COMMAND_STOP:
        if not memory.is_thread_tracked(event.interaction.channel_id):
            return
        await event.interaction.create_initial_response(
            hikari.ResponseType.MESSAGE_CREATE, f"OK, I will no longer participate in this thread. You can summon me back by pinging me 👋"
        )
        memory.stop_tracking_thread(event.interaction.channel_id)



async def load_channel_history(channel: hikari.GuildPublicThread) -> list[hikari.Message]:
    """

    """
    history = []
    async for message in channel.fetch_history():
        history.append(message)

    return history

@bot.listen()
async def new_guild_message(event: hikari.GuildMessageCreateEvent) -> None:
    """

    """
    if event.message.author.is_bot:
        return

    channel = await bot.rest.fetch_channel(event.message.channel_id)

    if channel.type not in (ChannelType.GUILD_PUBLIC_THREAD, ChannelType.GUILD_TEXT):
        return

    assert isinstance(channel, hikari.GuildPublicThread) or isinstance(channel, hikari.GuildTextChannel)

    reply_to = channel

    force_reply = bot.get_me().id in event.message.user_mentions

    if channel.type == ChannelType.GUILD_TEXT:
        # Should I start a thread to answer?
        decision: DecisionOutput = await decide_on_answer(event.message)
        if decision.answer or force_reply:
            # yes, start a new public thread
            reply_to = await bot.rest.create_thread(channel, ChannelType.GUILD_PUBLIC_THREAD,
                                                    decision.thread_title, auto_archive_duration=60)
            memory.start_tracking_thread(reply_to.id)
        else:
            return
    elif channel.type == ChannelType.GUILD_PUBLIC_THREAD:
        if not memory.is_thread_tracked(reply_to.id):
            # Not all threads are made for discussion with the bot.
            if force_reply:
                # But people can "invite" the bot to the discussion.
                memory.start_tracking_thread(reply_to.id)
            else:
                return

        # Ensure that ADK has the contents of this conversation.
        if not await session_exists(reply_to.id):
            print("No session here!")
            history = await load_channel_history(channel)
            await load_session_with_messages(reply_to.id, history)

    async with reply_to.trigger_typing():
        response = await reply_to_message(event.message, reply_to.id)
        answer = "".join(p.text for p in response.parts if p.text is not None)
        await reply_to.send(answer, reply=event.message if reply_to == channel else None)



def start_discord_engine():
    """
    Initializes the Discord bot async loop.
    """
    print('a')
    bot.run()
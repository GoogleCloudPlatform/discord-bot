# Copyright 2024 Google LLC
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

import argparse
import os
import sys

import hikari

from gemini_model import GeminiBot, VERTEX_TOS, GEMINI_TOS, GEMINI_API_KEY

bot = hikari.GatewayBot(
    token=os.getenv("DISCORD_BOT_TOKEN"),
    intents=hikari.Intents.ALL_GUILDS_UNPRIVILEGED | hikari.Intents.MESSAGE_CONTENT,
)

gemini_bot: GeminiBot


@bot.listen()
async def on_ready(event: hikari.StartedEvent):
    """
    Setting the GeminiBot instance as a global variable, so it can be accessed freely while
    handling the responses.
    """
    global gemini_bot
    gemini_bot = GeminiBot(bot.get_me())


@bot.listen()
async def new_guild_message(event: hikari.GuildMessageCreateEvent) -> None:
    """
    Forward all new guild messages to the GeminiBot for handling. This does not include
    Direct Messages that users can send to your bot.
    """

    await gemini_bot.handle_message(event)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="DiscordBot", description="A Discord bot using Gemini."
    )
    parser.add_argument(
        "--accept-tos",
        action="store_true",
        help="Use this flag to omit the question about Vertex AI and Gemini API ToS. "
        "By using this flag, you confirm that you've read and accepted Vertex AI and/or Gemini API ToS.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if not args.accept_tos:
        if GEMINI_API_KEY is None:
            response = input(
                f"Just to make sure, did you read and accept the Vertex AI API Terms of Service ( {VERTEX_TOS} )? [NO/yes] "
            )
        else:
            response = input(
                f"Just to make sure, did you read and accept the Gemini API Terms of Service ( {GEMINI_TOS} )? [NO/yes] "
            )

        if response.lower() != "yes":
            print(
                "Please make sure you read and accept the ToS before starting this bot."
            )
            sys.exit(1)
    bot.run()

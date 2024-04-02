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

import os
import sys

import hikari

import consent_management

_GEMINI_TOS = "https://ai.google.dev/terms"
_VERTEX_TOS = "https://developers.google.com/terms"

_CONSENT_REQUEST = """\
Before I can answer to you, I need you to consent to me sending your requests over to the AI model hosted by Google.

To give me your consent, please write a direct message to me saying "yes". 
"""

if os.getenv("GEMINI_API_KEY"):
    response = input(f"Just to make sure, did you read and accept Gemini API Terms of Service ( {_GEMINI_TOS} )? [NO/yes] ")
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
else:
    response = input(f"Just to make sure, did you read and accept the Vertex AI API Terms of Service ( {_VERTEX_TOS} )? [NO/yes] ")
    import vertexai.preview.generative_models as genai

if response.lower() != "yes":
    print("Please make sure you read and accept the ToS before starting this bot.")
    sys.exit(1)

model = genai.GenerativeModel(
    model_name="gemini-1.0-pro",
    generation_config=genai.GenerationConfig(max_output_tokens=1900)
)

bot = hikari.GatewayBot(token=os.getenv("DISCORD_BOT_TOKEN"))


@bot.listen()
async def private_message(event: hikari.DMMessageCreateEvent) -> None:
    """Check if the message says "yes", record the consent if yes."""
    if not event.is_human:
        return

    if event.message.content.lower() == "yes":
        consent_management.record_user_consent(event.message.author.id)
        await event.message.respond("Thank you, your consent has been recorded. I will now use AI to reply to your "
                                    "messages on the server.")


@bot.listen()
async def ping(event: hikari.GuildMessageCreateEvent) -> None:
    """If a non-bot user mentions your bot, forward the message to Gemini."""

    # Do not respond to bots nor webhooks pinging us, only user accounts
    if not event.is_human:
        return

    me = bot.get_me()

    if me.id in event.message.user_mentions_ids:
        if not consent_management.check_user_consent(event.message.author.id):
            # The response about consent will be visible only to the author of the message.
            await event.message.respond(_CONSENT_REQUEST, flags=hikari.MessageFlag.EPHEMERAL)
            return
        await event.get_channel().trigger_typing()
        result = await model.generate_content_async(event.message.content)
        await event.message.respond(result.text)

bot.run()

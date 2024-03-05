# GenAI Discord Bot

This repository hosts source code for a Discord bot using various Google Cloud AI-related products.

## Getting started

To run the bot, you will need access to [Gemini API](https://ai.google.dev/) or 
[Vertex AI API](https://cloud.google.com/vertex-ai/docs/python-sdk/use-vertex-ai-python-sdk-ref). In case of Gemini API,
you have to provide `GEMINI_API_KEY` as an environmental variable. For Vertex AI, the library will make use of the 
[pplication Default Credentials](https://cloud.google.com/docs/authentication/client-libraries).

The bot also requires you to set `DISCORD_BOT_TOKEN` environmental variable to contain the secret token generated
for your bot in the Discord Developer portal. Learn more about setting up your first application/bot from 
[Discord documentation](https://discord.com/developers/docs/getting-started).



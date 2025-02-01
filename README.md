# GenAI Discord Bot

This repository hosts source code for a Discord bot using various Google Cloud AI-related products.

## Getting started

To run the bot, you will need access to [Vertex AI API](https://cloud.google.com/vertex-ai/docs/python-sdk/use-vertex-ai-python-sdk-ref) or an API key from [Google AI Studio](https://aistudio.google.com/apikey).

If you decide to you the AI Studio API Key, set it as an environmental variable with `export GEMINI_API_KEY=your_key_here`.
The bot will try to use this variable to authenticate with the API. In case this variable is not set, the AI API calls
will be handled by Vertex AI using the [Application Default Credentials](https://cloud.google.com/docs/authentication/application-default-credentials).

The bot also requires you to set `DISCORD_BOT_TOKEN` environmental variable to contain the secret token generated
for your bot in the Discord Developer portal. Learn more about setting up your first application/bot from 
[Discord documentation](https://discord.com/developers/docs/getting-started).

## Vertex AI API Terms of Service

To execute the code of the bot, you need to have access to Vertex AI API. That's possible
only once you accept [Terms of Service](https://developers.google.com/terms). Please remember 
about those terms if you expose the bot to other users on your Discord server.

## Google AI Studio Terms of Service

By using the AI Studio API key, you accept the Gemini API [Terms of Service](https://ai.google.dev/gemini-api/terms).
Please remember about those terms if you expose the bot to other users on your Discord server.

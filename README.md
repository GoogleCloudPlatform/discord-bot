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


## Vertex AI API and Gemini API Terms of Service

To execute the code of the bot, you need to have access to Gemini API or Vertex AI API. In both cases, that's possible
only once you accept Terms of Service of the services you decide to use. Please remember about those terms if you
expose the bot to other users on your Discord server.

Gemini API ToS: https://ai.google.dev/terms
Vertex AI API ToS: https://developers.google.com/terms

It is your responsibility to respect the privacy of users interacting with your bot.
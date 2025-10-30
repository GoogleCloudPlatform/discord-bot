# ADK Discord Bot

This is a Discord bot built with the Google Agent Development Kit (ADK) and the `hikari` library.

## Prerequisites

- Python 3.10 or higher

## Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-repo/discord-bot.git
   cd discord-bot
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your environment variables:**

   Copy the `.env.template` file to a new file named `.env`:

   ```bash
   cp adk-bot/.env.template adk-bot/.env
   ```

   Open the `adk-bot/.env` file and replace the placeholder values with your actual Discord bot token and other settings.

## Running the Bot

Once you've completed the setup, you can run the bot with the following command:

```bash
cd adk-bot
python main.py
```

## Testing

You can run the tests by running `pytest` in the `adk-bot` folder.

## Building and Deploying

To build and deploy a new Docker Image, follow these steps:

```bash
PROJECT_ID=your-project-id
REPOSITORY=your-repository-name

# Execute in the root directory of this repository
docker build . -t europe-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/discord-bot:latest

# Remember to authenticate with gcloud before pushing
docker push europe-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/discord-bot:latest

```
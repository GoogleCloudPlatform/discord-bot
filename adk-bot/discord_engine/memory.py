import logging
import os
import pathlib

import platformdirs

MEMORY_DIR = pathlib.Path(os.getenv("DISCORD_BOT_MEMORY", pathlib.Path(platformdirs.user_data_dir("discord-bot"))))

logging.info(f"Bot memory located at: {MEMORY_DIR}")

TRACKED_THREADS = MEMORY_DIR / "tracked_threads"
TRACKED_THREADS.mkdir(parents=True, exist_ok=True)


def is_thread_tracked(channel_id: int):
    thread_file = TRACKED_THREADS / f"{channel_id}.thread"
    return thread_file.exists()


def start_tracking_thread(channel_id: int):
    thread_file = TRACKED_THREADS / f"{channel_id}.thread"
    thread_file.touch()
    pass

def stop_tracking_thread(channel_id: int):
    thread_file = TRACKED_THREADS / f"{channel_id}.thread"
    thread_file.unlink(missing_ok=True)
    pass
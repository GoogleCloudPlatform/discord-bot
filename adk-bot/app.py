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

from google.adk.app import App
from google.adk.apps.app import EventsCompactionConfig
from google.adk.context_caching import ContextCacheConfig

from agents import cache
from agents.initial_decision_maker import decision_agent
from agents.root_agent import root_agent
from config import APP_NAME, COMPACTION_INTERVAL, OVERLAP_SIZE


class BotApp(App):
    """Custom App class to handle pre-caching of URLs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cache.load_pre_cached_urls()


# Configure ContextCacheConfig
cache_config = ContextCacheConfig(
    cache_static_context_only=True,
)

# Configure EventsCompactionConfig
compaction_config = EventsCompactionConfig(
    compaction_interval=COMPACTION_INTERVAL,
    overlap_size=OVERLAP_SIZE,
)

# Create the App with the configured cache and compaction
root_app = BotApp(
    name=APP_NAME,
    root_agent=root_agent,
    context_cache_config=cache_config,
    events_compaction_config=compaction_config,
)

decision_app = BotApp(
    name=APP_NAME,
    root_agent=decision_agent,
    context_cache_config=cache_config,
    events_compaction_config=compaction_config,
)

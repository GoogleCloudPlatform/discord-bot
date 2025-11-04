# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from typing import Optional

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from google.adk.tools.vertex_ai_search_tool import VertexAiSearchTool
from google.genai import types

from agents.common import load_prompt
from config import VERTEXAI_SEARCH_ENGINE_ID


def citation_retrieval_after_model_callback(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> Optional[LlmResponse]:
  """Callback function to retrieve citations after model response is generated."""
  grounding_metadata = llm_response.grounding_metadata

  if not grounding_metadata:
    return None

  content = llm_response.content
  if not llm_response.content:
    return None

  parts = content.parts
  if not parts:
    return None

  # Add citations to the response as JSON objects.
  parts.append(types.Part(text="References:\n"))
  for grounding_chunk in grounding_metadata.grounding_chunks:
    retrieved_context = grounding_chunk.retrieved_context
    if not retrieved_context:
      continue

    citation = {
        "title": retrieved_context.title,
        "uri": retrieved_context.uri,
        "snippet": retrieved_context.text,
    }
    parts.append(types.Part(text=json.dumps(citation)))

  return LlmResponse(content=types.Content(parts=parts))


adk_knowledge_agent = LlmAgent(
    name="adk_knowledge_agent",
    description=(
        "Agent for performing Vertex AI Search to find ADK knowledge and all materials related to the Kaggle event."
    ),
    instruction=load_prompt("adk_search"),
    model="gemini-2.5-flash",
    tools=[VertexAiSearchTool(search_engine_id=VERTEXAI_SEARCH_ENGINE_ID)],
    after_model_callback=citation_retrieval_after_model_callback,
)
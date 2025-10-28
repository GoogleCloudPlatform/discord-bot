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

import io

import google.genai.types as types
from google.auth import default
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError
import time
from google.adk.agents import Agent

from config import DOCUMENT_READER_MODEL, DOCUMENT_READER_DESCRIPTION, DOCUMENT_READER_NAME
from .cache import CACHE_PATH
from .common import load_prompt

# The scope for read-only access to Google Drive
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

DOCUMENT_IDS = {
    'event_doc': '17LEZODotFphisk2FdOU-7RMTyljWITqKbXpljDGNDOc',
}

DOCUMENTS = {
    'event_doc': 'This document stores the information about the Kaggle event.'
}

assert set(DOCUMENT_IDS.keys()) == set(DOCUMENTS.keys())

DOCS_PATH = CACHE_PATH / "docs"
DOCS_PATH.mkdir(parents=True, exist_ok=True)

_DOWNLOAD_CACHE = {}
_CACHE_TTL = 60*30

def get_available_documents() -> dict:
    """
    Returns a list of available documents as a dictionary mapping document name to the description of document content.
    The contents of the documents can be retrieved with the get_document_as_pdf tool.

    Returns: dictionary with document descriptions.
    """
    print("Checking documents.")
    return DOCUMENTS


def get_document(doc_name: str) -> dict:
    """Retrieves the content of a given document.

    Args:
        doc_name: Name of the document to retrieve.
    """
    now = time.time()
    if doc_name in _DOWNLOAD_CACHE and (now - _DOWNLOAD_CACHE[doc_name]['time'] <= _CACHE_TTL):
        return {'status': "success", 'document': _DOWNLOAD_CACHE[doc_name]['part']}

    # Authenticate using Application Default Credentials
    creds, _ = default(scopes=SCOPES)

    try:
        service = build("drive", "v3", credentials=creds)

        request = service.files().export_media(
            fileId=DOCUMENT_IDS[doc_name], mimeType="text/plain"
        )
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            _, done = downloader.next_chunk()

        fh.seek(0)
        # part = types.Part.from_bytes(data=fh.read(), mime_type="application/pdf")
        part = types.Part.from_text(text=fh.read().decode())
        _DOWNLOAD_CACHE[doc_name] = {
            'time': time.time(),
            'part': part,
        }

        return {'status': "success", 'document': _DOWNLOAD_CACHE[doc_name]['part']}

    except HttpError as error:
        print(f"An error occurred: {error}")
        # Specifically handle the case where the service account might not have access
        if error.resp.status == 404:
            print("Error: The file was not found. Please ensure the Document ID is correct and that the "
                  "Service Account has at least 'Viewer' permissions on the Google Doc.")
        return {'status': 'error', 'error': f"Couldn't find the requested document: {doc_name}"}


document_reader = Agent(
    name=DOCUMENT_READER_NAME,
    model=DOCUMENT_READER_MODEL,
    description=DOCUMENT_READER_DESCRIPTION,
    instruction=(
        load_prompt('document_reader')
    ),
    tools=[get_available_documents, get_document]

)
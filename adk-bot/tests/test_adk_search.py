import dotenv
from google.genai.types import Content

from agents import reply_to_message
import pytest
from unittest.mock import Mock
import copy


basic_message = Mock()
basic_message.id = 122
basic_message.member.display_name = "User1"
basic_message.author.is_bot = False
basic_message.attachments = []
basic_message.content = "What is the second day of the event about?"

dotenv.load_dotenv()

@pytest.mark.asyncio
async def test_the_message():
    message = copy.copy(basic_message)
    res = await reply_to_message(message, 9001)
    assert isinstance(res, Content)
    assert res.parts
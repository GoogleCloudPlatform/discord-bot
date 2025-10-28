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
basic_message.content = "This is a test message"


@pytest.mark.asyncio
async def test_there_is_an_answer():
    message = copy.copy(basic_message)
    res = await reply_to_message(message, 9001)
    assert isinstance(res, Content)
    assert res.parts
import copy
from unittest.mock import Mock

import pytest

from agents.initial_decision_maker import decide_on_answer, DecisionOutput

basic_message = Mock()
basic_message.id = 123
basic_message.member.display_name = "User1"
basic_message.author.is_bot = False
basic_message.attachments = []
basic_message.content = "This is a test message"

@pytest.mark.asyncio
async def test_decision_answer_false():
    message = copy.copy(basic_message)
    res = await decide_on_answer(message)
    assert isinstance(res, DecisionOutput)
    assert res.answer is False


@pytest.mark.asyncio
async def test_decision_answer_true():
    message = copy.copy(basic_message)
    message.content = "How do I set up ADK to use Gemini 2.5 Pro?"
    res = await decide_on_answer(message)
    assert isinstance(res, DecisionOutput)
    assert res.answer is True
    assert res.thread_title
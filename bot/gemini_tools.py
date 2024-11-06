import datetime
import time

import pytz
from vertexai.generative_models import Part, Tool, FunctionDeclaration

from imagen import call_generate_image, generate_image_tool


def get_current_time(timezone: str) -> str:
    """
    Get the current time in a specified timezone.

    Args:
        timezone: The timezone for which you want to check the time. For example: Europe/Warsaw.

    Returns:
        Current time in a given timezone, formatted in the YYYY-MM-DD HH:MM:SS format.
    """
    tz = pytz.timezone(timezone)
    tz_now = datetime.datetime.now(tz)
    return tz_now.strftime("%Y-%m-%d %H:%M:%S")


def call_get_current_time(call_part: Part) -> (str, None):
    assert call_part.function_call.name == "get_current_time"
    return get_current_time(call_part.function_call.args["timezone"]), None


def noop() -> str:
    """
    Do nothing, just allows you to normally reply to users, but in the function calling mode. Use this, when the user is not asking you to use any other function.

    :return:
        Nothing of value, not important.
    """
    time.sleep(0.5)
    return "Nothing happens"


def call_noop(call_part: Part) -> (str, None):
    return noop(), None

# The tool calling functions need to return a string, as a response to the LLM
# And optionally an attachment that later needs to be handled and added to the response message.
TOOL_CALLING = {
    "get_current_time": call_get_current_time,
    "generate_image": call_generate_image,
    "noop": call_noop,
}

_TOOLS_NO_NOOP = [
    generate_image_tool,
    FunctionDeclaration.from_func(get_current_time),
]

_TOOLS_ALL = _TOOLS_NO_NOOP + [FunctionDeclaration.from_func(noop)]

TOOLS_NO_NOOP = [Tool(_TOOLS_NO_NOOP)]

TOOLS = [Tool(_TOOLS_ALL)]

IMG_TOOLS_NOOP = [Tool([generate_image_tool, FunctionDeclaration.from_func(noop)])]

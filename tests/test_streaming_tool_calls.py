"""Streaming function calling must produce chunks OpenAI clients can consume.

OpenAI's streaming spec requires an ``index`` on every tool_call delta and a
final ``finish_reason`` of ``tool_calls``. Agent clients (Hermes, the openai
SDK's stream helpers) accumulate tool calls by index and only run them when
the stream ends with that finish reason.
"""

import asyncio
import json

import pytest
from openai.types.chat import ChatCompletionChunk

from src import main as main_mod
from src.models import ChatCompletionRequest, Message

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }
]


def _fake_completion(text):
    async def fake_run_completion(**kwargs):
        yield {"content": [{"type": "text", "text": text}]}

    return fake_run_completion


def _stream(monkeypatch, text, tools=TOOLS):
    monkeypatch.setattr(main_mod.claude_cli, "run_completion", _fake_completion(text))
    request = ChatCompletionRequest(
        model="claude-sonnet-4-6",
        messages=[Message(role="user", content="weather in Berlin?")],
        stream=True,
        tools=tools,
    )

    async def collect():
        return [c async for c in main_mod.generate_streaming_response(request, "req-tc")]

    chunks = []
    for line in "".join(asyncio.run(collect())).splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            chunks.append(ChatCompletionChunk.model_validate(json.loads(line[6:])))
    return chunks


def test_tool_call_deltas_have_index_and_finish_reason_tool_calls(monkeypatch):
    text = (
        "```tool_calls\n"
        '[{"name": "get_weather", "arguments": {"city": "Berlin"}},'
        ' {"name": "get_weather", "arguments": {"city": "Paris"}}]\n'
        "```"
    )
    chunks = _stream(monkeypatch, text)

    tool_deltas = [
        tc for c in chunks for ch in c.choices for tc in (ch.delta.tool_calls or [])
    ]
    assert [tc.index for tc in tool_deltas] == [0, 1]
    assert [json.loads(tc.function.arguments)["city"] for tc in tool_deltas] == [
        "Berlin",
        "Paris",
    ]
    assert all(tc.id and tc.type == "function" for tc in tool_deltas)

    finish = [ch.finish_reason for c in chunks for ch in c.choices if ch.finish_reason]
    assert finish == ["tool_calls"]


def test_plain_text_reply_with_tools_still_finishes_with_stop(monkeypatch):
    chunks = _stream(monkeypatch, "It is sunny in Berlin.")

    assert not any(ch.delta.tool_calls for c in chunks for ch in c.choices)
    finish = [ch.finish_reason for c in chunks for ch in c.choices if ch.finish_reason]
    assert finish == ["stop"]

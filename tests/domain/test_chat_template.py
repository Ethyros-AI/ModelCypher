# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from modelcypher.core.domain.chat_template import ChatMessage, ChatTemplate, ChatTemplateEngine


def test_chat_template_detect_routes_common_model_names() -> None:
    assert ChatTemplate.detect("Llama-3.2-vision") == ChatTemplate.llama3Vision
    assert ChatTemplate.detect("Qwen-3-14B") == ChatTemplate.qwen3
    assert ChatTemplate.detect("Qwen-2-VL") == ChatTemplate.qwen2VL
    assert ChatTemplate.detect("gemma-2-9b") == ChatTemplate.gemma2
    assert ChatTemplate.detect("deepseek-r1") == ChatTemplate.deepseekR1
    assert ChatTemplate.detect("unknown-model") == ChatTemplate.chatml


def test_chat_template_format_instruction_alpaca_handles_optional_input() -> None:
    with_input = ChatTemplate.alpaca.format_instruction(
        instruction="Solve this",
        input_text="x=2",
        output="result",
    )
    without_input = ChatTemplate.alpaca.format_instruction(
        instruction="Solve this",
        input_text=None,
        output="result",
    )

    assert "### Instruction:" in with_input
    assert "### Input:" in with_input
    assert "x=2" in with_input
    assert "### Response:" in with_input

    assert "### Input:" not in without_input
    assert without_input.endswith("### Response:\nresult")


def test_chat_template_format_messages_mistral_inlines_system_prompt() -> None:
    messages = [
        ChatMessage(role="system", content="Be concise."),
        ChatMessage(role="user", content="Hello"),
        ChatMessage(role="assistant", content="Hi"),
        ChatMessage(role="user", content="How are you?"),
        ChatMessage(role="assistant", content="Great."),
    ]

    rendered = ChatTemplate.mistral.format_messages(messages)

    assert rendered.startswith("<s>[INST] Be concise.\n\nHello [/INST] Hi</s>")
    assert "<s>[INST] How are you? [/INST] Great.</s>" in rendered


def test_chat_template_engine_prefers_explicit_template_over_detected_model() -> None:
    messages = [ChatMessage(role="user", content="Ping"), ChatMessage(role="assistant", content="Pong")]

    rendered = ChatTemplateEngine.apply_template(
        messages=messages,
        model_name="llama-3",  # Would detect llama3 if template not provided.
        template=ChatTemplate.vicuna,
    )

    assert rendered == "USER: Ping\nASSISTANT: Pong"


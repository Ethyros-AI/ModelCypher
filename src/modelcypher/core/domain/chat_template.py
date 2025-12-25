# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModelCypher is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ModelCypher.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


class ChatTemplate(str, Enum):
    llama3 = "llama3"
    llama2 = "llama2"
    llama3Vision = "llama3_vision"
    qwen2 = "qwen2"
    qwen3 = "qwen3"
    qwen2VL = "qwen2vl"
    gemma2 = "gemma2"
    gemma3 = "gemma3"
    gemma3n = "gemma3n"
    mistral = "mistral"
    mistralV3 = "mistral_v3"
    pixtral = "pixtral"
    phi3 = "phi3"
    phi4 = "phi4"
    commandR = "command_r"
    commandRV2 = "command_r_v2"
    deepseek = "deepseek"
    deepseekR1 = "deepseek_r1"
    granite = "granite"
    gptOSS = "gpt_oss"
    alpaca = "alpaca"
    vicuna = "vicuna"
    zephyr = "zephyr"
    chatml = "chatml"

    def format_instruction(
        self,
        instruction: str,
        input_text: str | None,
        output: str,
        system_prompt: str = "[Environment context.]",
    ) -> str:
        if self in {ChatTemplate.llama3, ChatTemplate.llama3Vision}:
            result = (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"{system_prompt}<|eot_id|>"
            )
            result += f"<|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|>"
            result += f"<|start_header_id|>assistant<|end_header_id|>\n\n{output}<|eot_id|>"
            return result
        if self == ChatTemplate.llama2:
            return f"<s>[INST] {instruction} [/INST] {output} </s>"
        if self in {ChatTemplate.qwen2, ChatTemplate.qwen3, ChatTemplate.qwen2VL}:
            result = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            result += f"<|im_start|>user\n{instruction}<|im_end|>\n"
            result += f"<|im_start|>assistant\n{output}<|im_end|>"
            return result
        if self in {ChatTemplate.gemma2, ChatTemplate.gemma3, ChatTemplate.gemma3n}:
            result = f"<bos><start_of_turn>user\n{instruction}<end_of_turn>\n"
            result += f"<start_of_turn>model\n{output}<end_of_turn>"
            return result
        if self in {ChatTemplate.mistral, ChatTemplate.mistralV3, ChatTemplate.pixtral}:
            return f"<s>[INST] {instruction} [/INST] {output}</s>"
        if self in {ChatTemplate.phi3, ChatTemplate.phi4}:
            result = f"<|user|>\n{instruction}<|end|>\n"
            result += f"<|assistant|>\n{output}<|end|>"
            return result
        if self in {ChatTemplate.commandR, ChatTemplate.commandRV2}:
            result = "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>"
            result += f"{instruction}<|END_OF_TURN_TOKEN|>"
            result += "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
            result += f"{output}<|END_OF_TURN_TOKEN|>"
            return result
        if self in {ChatTemplate.deepseek, ChatTemplate.deepseekR1}:
            return f"User: {instruction}\n\nAssistant: {output}"
        if self == ChatTemplate.granite:
            return f"<|user|>\n{instruction}\n<|assistant|>\n{output}"
        if self == ChatTemplate.gptOSS:
            result = f"<|im_start|>user\n{instruction}<|im_end|>\n"
            result += f"<|im_start|>assistant\n{output}<|im_end|>"
            return result
        if self == ChatTemplate.alpaca:
            result = f"### Instruction:\n{instruction}\n\n"
            if input_text:
                result += f"### Input:\n{input_text}\n\n"
            result += f"### Response:\n{output}"
            return result
        if self == ChatTemplate.vicuna:
            return f"USER: {instruction}\nASSISTANT: {output}"
        if self == ChatTemplate.zephyr:
            result = f"<|system|>\n{system_prompt}</s>\n"
            result += f"<|user|>\n{instruction}</s>\n"
            result += f"<|assistant|>\n{output}</s>"
            return result
        if self == ChatTemplate.chatml:
            result = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            result += f"<|im_start|>user\n{instruction}<|im_end|>\n"
            result += f"<|im_start|>assistant\n{output}<|im_end|>"
            return result
        return f"{instruction}\n\n{output}"

    def format_messages(self, messages: list[ChatMessage]) -> str:
        if self in {ChatTemplate.mistral, ChatTemplate.mistralV3, ChatTemplate.pixtral}:
            return self._format_mistral(messages)
        if self in {ChatTemplate.llama3, ChatTemplate.llama3Vision}:
            return self._format_llama3(messages)
        if self == ChatTemplate.llama2:
            return self._format_llama2(messages)
        if self in {ChatTemplate.phi3, ChatTemplate.phi4}:
            return self._format_phi(messages)
        if self in {ChatTemplate.qwen2, ChatTemplate.qwen3, ChatTemplate.qwen2VL}:
            return self._format_qwen(messages)
        if self in {ChatTemplate.gemma2, ChatTemplate.gemma3, ChatTemplate.gemma3n}:
            return self._format_gemma(messages)
        if self in {ChatTemplate.commandR, ChatTemplate.commandRV2}:
            return self._format_command_r(messages)
        if self in {ChatTemplate.deepseek, ChatTemplate.deepseekR1}:
            return self._format_deepseek(messages)
        if self == ChatTemplate.granite:
            return self._format_granite(messages)
        if self == ChatTemplate.gptOSS:
            return self._format_gpt_oss(messages)
        if self == ChatTemplate.alpaca:
            return self._format_alpaca(messages)
        if self == ChatTemplate.vicuna:
            return self._format_vicuna(messages)
        if self == ChatTemplate.zephyr:
            return self._format_zephyr(messages)
        if self == ChatTemplate.chatml:
            return self._format_chatml(messages)
        return "\n".join(f"{message.role}: {message.content}" for message in messages)

    def _format_mistral(self, messages: list[ChatMessage]) -> str:
        result = ""
        system_prompt = None
        if messages and messages[0].role == "system":
            system_prompt = messages[0].content
        conversation = [message for message in messages if message.role != "system"]
        for index, message in enumerate(conversation):
            if message.role == "user":
                user_content = message.content
                if index == 0 and system_prompt is not None:
                    user_content = f"{system_prompt}\n\n{message.content}"
                result += f"<s>[INST] {user_content} [/INST]"
            elif message.role == "assistant":
                result += f" {message.content}</s>"
        return result

    def _format_llama3(self, messages: list[ChatMessage]) -> str:
        result = "<|begin_of_text|>"
        for message in messages:
            result += (
                f"<|start_header_id|>{message.role}<|end_header_id|>\n\n{message.content}<|eot_id|>"
            )
        return result

    def _format_llama2(self, messages: list[ChatMessage]) -> str:
        result = ""
        for message in messages:
            if message.role == "user":
                result += f"<s>[INST] {message.content} [/INST]"
            elif message.role == "assistant":
                result += f" {message.content}</s>"
        return result

    def _format_phi(self, messages: list[ChatMessage]) -> str:
        result = ""
        for message in messages:
            result += f"<|{message.role}|>\n{message.content}<|end|>\n"
        return result

    def _format_qwen(self, messages: list[ChatMessage]) -> str:
        result = ""
        for message in messages:
            result += f"<|im_start|>{message.role}\n{message.content}<|im_end|>\n"
        return result

    def _format_gemma(self, messages: list[ChatMessage]) -> str:
        result = ""
        for message in messages:
            if message.role == "user":
                result += f"<start_of_turn>user\n{message.content}<end_of_turn>\n"
            elif message.role == "assistant":
                result += f"<start_of_turn>model\n{message.content}<end_of_turn>\n"
        return result

    def _format_command_r(self, messages: list[ChatMessage]) -> str:
        result = ""
        for message in messages:
            role = message.role.upper()
            result += (
                f"<|START_OF_TURN_TOKEN|><|{role}_TOKEN|>{message.content}<|END_OF_TURN_TOKEN|>"
            )
        return result

    def _format_deepseek(self, messages: list[ChatMessage]) -> str:
        return "\n\n".join(
            f"{message.role.capitalize()}: {message.content}" for message in messages
        )

    def _format_granite(self, messages: list[ChatMessage]) -> str:
        return "\n".join(f"<|{message.role}|>\n{message.content}" for message in messages)

    def _format_gpt_oss(self, messages: list[ChatMessage]) -> str:
        result = ""
        for message in messages:
            result += f"<|im_start|>{message.role}\n{message.content}<|im_end|>\n"
        return result

    def _format_alpaca(self, messages: list[ChatMessage]) -> str:
        return "\n\n".join(
            f"### {message.role.capitalize()}:\n{message.content}" for message in messages
        )

    def _format_vicuna(self, messages: list[ChatMessage]) -> str:
        return "\n".join(f"{message.role.upper()}: {message.content}" for message in messages)

    def _format_zephyr(self, messages: list[ChatMessage]) -> str:
        result = ""
        for message in messages:
            result += f"<|{message.role}|>\n{message.content}</s>\n"
        return result

    def _format_chatml(self, messages: list[ChatMessage]) -> str:
        result = ""
        for message in messages:
            result += f"<|im_start|>{message.role}\n{message.content}<|im_end|>\n"
        return result

    @staticmethod
    def detect(model_name: str) -> "ChatTemplate":
        name = model_name.lower()

        if "llama-3" in name or "llama3" in name:
            if "vision" in name or "vlm" in name:
                return ChatTemplate.llama3Vision
            return ChatTemplate.llama3
        if "llama-2" in name or "llama2" in name:
            return ChatTemplate.llama2

        if "qwen2vl" in name or "qwen-2-vl" in name:
            return ChatTemplate.qwen2VL
        if "qwen3" in name or "qwen-3" in name:
            return ChatTemplate.qwen3
        if "qwen2" in name or "qwen-2" in name:
            return ChatTemplate.qwen2

        if "gemma-3n" in name or "gemma3n" in name:
            return ChatTemplate.gemma3n
        if "gemma-3" in name or "gemma3" in name:
            return ChatTemplate.gemma3
        if "gemma-2" in name or "gemma2" in name or "gemma" in name:
            return ChatTemplate.gemma2

        if "pixtral" in name:
            return ChatTemplate.pixtral
        if "mistral" in name and "v3" in name:
            return ChatTemplate.mistralV3
        if "mistral" in name or "mixtral" in name:
            return ChatTemplate.mistral

        if "phi-4" in name or "phi4" in name:
            return ChatTemplate.phi4
        if "phi-3" in name or "phi3" in name:
            return ChatTemplate.phi3

        if "command-r" in name and "v2" in name:
            return ChatTemplate.commandRV2
        if "command-r" in name or "command_r" in name:
            return ChatTemplate.commandR

        if "deepseek-r1" in name or "deepseek_r1" in name:
            return ChatTemplate.deepseekR1
        if "deepseek" in name:
            return ChatTemplate.deepseek

        if "granite" in name:
            return ChatTemplate.granite

        if "gpt-oss" in name:
            return ChatTemplate.gptOSS

        if "alpaca" in name:
            return ChatTemplate.alpaca
        if "vicuna" in name:
            return ChatTemplate.vicuna
        if "zephyr" in name:
            return ChatTemplate.zephyr

        return ChatTemplate.chatml


class ChatTemplateEngine:
    @staticmethod
    def apply_template(
        messages: list[ChatMessage],
        model_name: str | None = None,
        template: ChatTemplate | None = None,
    ) -> str:
        selected = template or ChatTemplate.detect(model_name or "")
        return selected.format_messages(messages)

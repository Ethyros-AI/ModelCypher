from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def strip_swift_comments(text: str) -> str:
    result: list[str] = []
    in_string = False
    in_line_comment = False
    in_block_comment = False
    escape = False
    i = 0
    while i < len(text):
        ch = text[i]
        next_two = text[i : i + 2]
        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
                result.append(ch)
            i += 1
            continue
        if in_block_comment:
            if next_two == "*/":
                in_block_comment = False
                i += 2
            else:
                i += 1
            continue
        if in_string:
            result.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if next_two == "//":
            in_line_comment = True
            i += 2
            continue
        if next_two == "/*":
            in_block_comment = True
            i += 2
            continue
        if ch == '"':
            in_string = True
            result.append(ch)
            i += 1
            continue
        result.append(ch)
        i += 1
    return "".join(result)


def find_matching(text: str, start: int, open_ch: str, close_ch: str) -> int:
    depth = 0
    in_string = False
    escape = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            i += 1
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError("No matching bracket found")


def extract_list_block(text: str, marker: str) -> str:
    idx = text.find(marker)
    if idx == -1:
        raise ValueError(f"Marker not found: {marker}")
    assign_idx = text.find("=", idx)
    if assign_idx == -1:
        raise ValueError(f"Assignment not found for marker: {marker}")
    list_start = text.find("[", assign_idx)
    if list_start == -1:
        raise ValueError(f"List start not found for marker: {marker}")
    list_end = find_matching(text, list_start, "[", "]")
    return text[list_start + 1 : list_end]


def extract_blocks(list_text: str, prefixes: list[str]) -> list[str]:
    blocks: list[str] = []
    i = 0
    while i < len(list_text):
        matched = None
        for prefix in prefixes:
            if list_text.startswith(prefix, i):
                matched = prefix
                break
        if matched is None:
            i += 1
            continue
        start_paren = list_text.find("(", i + len(matched) - 1)
        if start_paren == -1:
            i += len(matched)
            continue
        end_paren = find_matching(list_text, start_paren, "(", ")")
        blocks.append(list_text[start_paren + 1 : end_paren])
        i = end_paren + 1
    return blocks


def split_top_level(text: str, delimiter: str) -> list[str]:
    items: list[str] = []
    current: list[str] = []
    in_string = False
    escape = False
    depth_paren = 0
    depth_bracket = 0
    for ch in text:
        if in_string:
            current.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            current.append(ch)
            continue
        if ch == "(":
            depth_paren += 1
            current.append(ch)
            continue
        if ch == ")":
            depth_paren -= 1
            current.append(ch)
            continue
        if ch == "[":
            depth_bracket += 1
            current.append(ch)
            continue
        if ch == "]":
            depth_bracket -= 1
            current.append(ch)
            continue
        if ch == delimiter and depth_paren == 0 and depth_bracket == 0:
            item = "".join(current).strip()
            if item:
                items.append(item)
            current = []
            continue
        current.append(ch)
    tail = "".join(current).strip()
    if tail:
        items.append(tail)
    return items


def split_key_value(text: str) -> tuple[str, str]:
    in_string = False
    escape = False
    depth_paren = 0
    depth_bracket = 0
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "(":
            depth_paren += 1
            continue
        if ch == ")":
            depth_paren -= 1
            continue
        if ch == "[":
            depth_bracket += 1
            continue
        if ch == "]":
            depth_bracket -= 1
            continue
        if ch == ":" and depth_paren == 0 and depth_bracket == 0:
            return text[:i].strip(), text[i + 1 :].strip()
    raise ValueError(f"No key/value separator in: {text}")


def parse_swift_string(text: str) -> str:
    if not text.startswith('"') or not text.endswith('"'):
        raise ValueError(f"Not a Swift string: {text}")
    body = text[1:-1]
    result: list[str] = []
    i = 0
    while i < len(body):
        ch = body[i]
        if ch == "\\" and i + 1 < len(body):
            nxt = body[i + 1]
            if nxt in ["\\", '"', "n", "t", "r"]:
                if nxt == "n":
                    result.append("\n")
                elif nxt == "t":
                    result.append("\t")
                elif nxt == "r":
                    result.append("\r")
                else:
                    result.append(nxt)
                i += 2
                continue
        result.append(ch)
        i += 1
    return "".join(result)


def parse_value(text: str) -> Any:
    text = text.strip()
    if not text:
        return None
    if text.startswith('"'):
        return parse_swift_string(text)
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        items = split_top_level(inner, ",")
        return [parse_value(item) for item in items]
    if text.startswith(".init") or text.startswith("EnrichedPrime") or text.startswith("LanguageTexts"):
        return parse_init(text)
    if text.startswith("."):
        return text[1:]
    if text == "nil":
        return None
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"-?\d+\.\d+", text):
        return float(text)
    return text


def parse_init(text: str) -> dict[str, Any]:
    start = text.find("(")
    end = text.rfind(")")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Invalid init block: {text}")
    inner = text[start + 1 : end]
    args = split_top_level(inner, ",")
    data: dict[str, Any] = {}
    for arg in args:
        if not arg:
            continue
        key, value = split_key_value(arg)
        data[key] = parse_value(value)
    return data


def parse_semantic_primes(swift_path: Path) -> list[dict[str, Any]]:
    raw = swift_path.read_text(encoding="utf-8")
    text = strip_swift_comments(raw)
    list_text = extract_list_block(text, "public static let english2014")
    blocks = extract_blocks(list_text, [".init("])
    primes = [parse_init(".init(" + block + ")") for block in blocks]
    return primes


def parse_enriched_primes(swift_path: Path) -> list[dict[str, Any]]:
    raw = swift_path.read_text(encoding="utf-8")
    text = strip_swift_comments(raw)
    list_text = extract_list_block(text, "public static let enriched")
    blocks = extract_blocks(list_text, ["EnrichedPrime("])
    primes = [parse_init("EnrichedPrime(" + block + ")") for block in blocks]
    return primes


def parse_multilingual_inventory(swift_path: Path, marker: str) -> dict[str, Any]:
    raw = swift_path.read_text(encoding="utf-8")
    text = strip_swift_comments(raw)
    idx = text.find(marker)
    if idx == -1:
        raise ValueError(f"Marker not found: {marker}")
    primes_idx = text.find("let primes", idx)
    if primes_idx == -1:
        raise ValueError(f"Prime list not found for marker: {marker}")
    eq_idx = text.find("=", primes_idx)
    if eq_idx == -1:
        raise ValueError(f"Prime list assignment not found for marker: {marker}")
    list_start = text.find("[", eq_idx)
    if list_start == -1:
        raise ValueError(f"Prime list start not found for marker: {marker}")
    list_end = find_matching(text, list_start, "[", "]")
    list_text = text[list_start + 1 : list_end]
    blocks = extract_blocks(list_text, [".init("])
    primes = [parse_init(".init(" + block + ")") for block in blocks]
    meta = parse_multilingual_meta(text, marker)
    meta["primes"] = primes
    return meta


def parse_multilingual_meta(text: str, marker: str) -> dict[str, Any]:
    idx = text.find(marker)
    if idx == -1:
        raise ValueError(f"Marker not found for meta: {marker}")
    return_idx = text.find("return SemanticPrimeMultilingualInventory", idx)
    if return_idx == -1:
        raise ValueError(f"Return block not found for meta: {marker}")
    start_paren = text.find("(", return_idx)
    end_paren = find_matching(text, start_paren, "(", ")")
    inner = text[start_paren + 1 : end_paren]
    args = split_top_level(inner, ",")
    meta: dict[str, Any] = {}
    for arg in args:
        if not arg:
            continue
        key, value = split_key_value(arg)
        if key == "primes":
            continue
        meta[key] = parse_value(value)
    return meta


def parse_computational_gates(swift_path: Path, marker: str) -> list[dict[str, Any]]:
    raw = swift_path.read_text(encoding="utf-8")
    text = strip_swift_comments(raw)
    list_text = extract_list_block(text, marker)
    blocks = extract_blocks(list_text, [".init("])
    gates = [parse_init(".init(" + block + ")") for block in blocks]
    return gates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--semantic-prime-atlas",
        required=True,
        help="Path to the Swift semantic prime atlas source file",
    )
    parser.add_argument(
        "--semantic-prime-frames",
        required=True,
        help="Path to the Swift semantic prime frames source file",
    )
    parser.add_argument(
        "--semantic-prime-multilingual",
        required=True,
        help="Path to the Swift semantic prime multilingual inventory source file",
    )
    parser.add_argument(
        "--computational-gates",
        required=True,
        help="Path to the Swift computational gate atlas source file",
    )
    parser.add_argument("--output", required=True, help="Output directory for JSON data")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    semantic_prime_path = Path(args.semantic_prime_atlas)
    semantic_frames_path = Path(args.semantic_prime_frames)
    multilingual_path = Path(args.semantic_prime_multilingual)
    gates_path = Path(args.computational_gates)

    semantic_primes = parse_semantic_primes(semantic_prime_path)
    enriched_primes = parse_enriched_primes(semantic_frames_path)
    core_european = parse_multilingual_inventory(multilingual_path, "public static let coreEuropean")
    global_diverse = parse_multilingual_inventory(multilingual_path, "public static let globalDiverse")
    core_gates = parse_computational_gates(gates_path, "public static let coreGates")
    composite_gates = parse_computational_gates(gates_path, "public static let compositeGates")

    (output / "semantic_primes.json").write_text(
        json.dumps({"english2014": semantic_primes}, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    (output / "semantic_prime_frames.json").write_text(
        json.dumps({"enriched": enriched_primes}, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    (output / "semantic_prime_multilingual.json").write_text(
        json.dumps({"coreEuropean": core_european, "globalDiverse": global_diverse}, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    (output / "computational_gates.json").write_text(
        json.dumps({"coreGates": core_gates, "compositeGates": composite_gates}, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

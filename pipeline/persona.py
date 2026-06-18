"""User persona extraction (Phase 3).

Extracts habits, personal facts, personality traits, and communication style
from conversation messages. Only records facts supported by explicit evidence
in User 1 messages (the primary speaker whose persona we profile).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from tqdm.auto import tqdm

LOGGER = logging.getLogger(__name__)

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)

# Each rule: label -> list of regex patterns that must match in message text.
HABIT_RULES: Dict[str, List[str]] = {
    "late sleeper": [
        r"\b(stay up late|night owl|up late|can't sleep|insomnia|go to bed late)\b",
    ],
    "early riser": [
        r"\b(wake up early|morning person|up at \d|early morning)\b",
    ],
    "coffee drinker": [
        r"\b(coffee|espresso|latte|cappuccino|americano)\b",
    ],
    "tea drinker": [
        r"\b(tea|chai|matcha|green tea)\b",
    ],
    "runner": [
        r"\b(running|go for a run|jogging|marathon|5k|10k)\b",
    ],
    "gym enthusiast": [
        r"\b(gym|workout|lift weights|exercise routine|hit the gym)\b",
    ],
    "reader": [
        r"\b(reading a book|love to read|finished (?:a |the )?book|avid reader)\b",
    ],
    "home cook": [
        r"\b(cook(?:ing)? at home|home-cooked|meal prep|recipe)\b",
    ],
    "vegetarian": [
        r"\b(vegetarian|don't eat meat|plant[- ]based)\b",
    ],
    "meditation practitioner": [
        r"\b(meditat(?:e|ion|ing)|mindfulness)\b",
    ],
}

PERSONAL_FACT_RULES: Dict[str, List[str]] = {
    "in a relationship": [
        r"\b(boyfriend|girlfriend|partner|dating|in a relationship)\b",
    ],
    "married": [
        r"\b(married|husband|wife|spouse|wedding)\b",
    ],
    "has children": [
        r"\b(my (?:son|daughter|kid|child|children|baby))\b",
    ],
    "pursuing education": [
        r"\b(college|university|studying|degree|grad school|student)\b",
    ],
    "working professional": [
        r"\b(my job|at work|coworker|office|employer|career)\b",
    ],
    "recently moved or relocating": [
        r"\b(moving to|moved to|relocat(?:e|ing)|new city|new apartment)\b",
    ],
    "interested in culinary career": [
        r"\b(culinary|chef|cooking career|restaurant|food industry)\b",
    ],
    "lives in or mentions Portland": [
        r"\b(portland(?:,?\s*oregon)?)\b",
    ],
    "lives in or mentions California": [
        r"\b(california|los angeles|san francisco|bay area)\b",
    ],
    "lives in or mentions New York": [
        r"\b(new york(?: city)?|nyc|brooklyn|manhattan)\b",
    ],
}

PERSONALITY_RULES: Dict[str, List[str]] = {
    "enthusiastic": [
        r"\b(excited|can't wait|amazing|awesome|love (?:it|this)|so happy)\b",
        r"!{2,}",
    ],
    "humorous": [
        r"\b(haha|lol|lmao|joke|funny)\b",
    ],
    "optimistic": [
        r"\b(looking forward|hopeful|positive|bright side|things will)\b",
    ],
    "ambitious": [
        r"\b(dream|goal|pursu(?:e|ing)|ambition|aspir(?:e|ing))\b",
    ],
    "emotional": [
        r"\b(feel(?:ing)?|emotional|upset|sad|cry|crying|heartbroken)\b",
    ],
    "supportive": [
        r"\b(proud of you|here for you|support you|you got this|believe in you)\b",
    ],
    "curious": [
        r"\b(curious|wondering|how does|what do you think|tell me more)\?",
    ],
    "serious": [
        r"\b(important|serious(?:ly)?|concerned|need to talk|honestly)\b",
    ],
}

CASUAL_MARKERS = re.compile(
    r"\b(yeah|yep|nope|gonna|wanna|kinda|sorta|lol|haha|btw|omg)\b",
    re.IGNORECASE,
)
FORMAL_MARKERS = re.compile(
    r"\b(however|therefore|furthermore|regarding|appreciate|sincerely)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PersonaConfig:
    messages_path: Path
    output_path: Path
    target_speaker: str
    max_evidence_per_item: int
    min_evidence_count: int
    limit_messages: int | None


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def load_messages(messages_path: Path) -> List[Dict[str, Any]]:
    if not messages_path.exists():
        raise FileNotFoundError(f"messages.json not found: {messages_path}")

    with messages_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError("messages.json must contain a JSON list.")

    payload.sort(key=lambda item: item["message_id"])
    return payload


def save_json(payload: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def compile_rules(rules: Dict[str, List[str]]) -> Dict[str, List[re.Pattern[str]]]:
    return {
        label: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        for label, patterns in rules.items()
    }


def message_matches_any(text: str, patterns: Sequence[re.Pattern[str]]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def extract_with_evidence(
    messages: Sequence[Dict[str, Any]],
    rules: Dict[str, List[str]],
    *,
    max_evidence: int,
    min_evidence: int,
) -> List[Dict[str, Any]]:
    compiled = compile_rules(rules)
    results: List[Dict[str, Any]] = []

    for label, patterns in compiled.items():
        evidence: List[Dict[str, Any]] = []
        for message in messages:
            text = str(message.get("text", "")).strip()
            if not text:
                continue
            if message_matches_any(text, patterns):
                evidence.append(
                    {
                        "message_id": int(message["message_id"]),
                        "conversation_id": int(message["conversation_id"]),
                        "text": text[:300],
                    }
                )
                if len(evidence) >= max_evidence:
                    break

        if len(evidence) >= min_evidence:
            results.append({"label": label, "evidence": evidence, "count": len(evidence)})

    results.sort(key=lambda item: item["count"], reverse=True)
    return results


def categorize_length(avg_length: float) -> str:
    if avg_length < 35:
        return "short"
    if avg_length < 90:
        return "medium"
    return "long"


def categorize_punctuation(exclamation_rate: float, question_rate: float) -> str:
    if exclamation_rate > 0.15:
        return "expressive"
    if question_rate > 0.20:
        return "inquisitive"
    if exclamation_rate < 0.03 and question_rate < 0.05:
        return "minimal"
    return "moderate"


def infer_tone(casual_rate: float, formal_rate: float) -> str:
    if casual_rate > formal_rate * 2 and casual_rate > 0.05:
        return "casual"
    if formal_rate > casual_rate * 2 and formal_rate > 0.02:
        return "formal"
    return "neutral"


def analyze_communication_style(
    messages: Sequence[Dict[str, Any]],
    *,
    sample_size: int = 5,
) -> Dict[str, Any]:
    if not messages:
        return {
            "avg_message_length": 0,
            "avg_length_category": "unknown",
            "emoji_usage": False,
            "emoji_rate": 0.0,
            "exclamation_rate": 0.0,
            "question_rate": 0.0,
            "punctuation_style": "unknown",
            "tone": "unknown",
            "evidence_samples": [],
        }

    lengths: List[int] = []
    emoji_messages = 0
    exclamation_messages = 0
    question_messages = 0
    casual_hits = 0
    formal_hits = 0

    for message in messages:
        text = str(message.get("text", "")).strip()
        if not text:
            continue

        lengths.append(len(text))
        if EMOJI_PATTERN.search(text):
            emoji_messages += 1
        if "!" in text:
            exclamation_messages += 1
        if "?" in text:
            question_messages += 1
        if CASUAL_MARKERS.search(text):
            casual_hits += 1
        if FORMAL_MARKERS.search(text):
            formal_hits += 1

    total = len(lengths)
    avg_length = sum(lengths) / total
    emoji_rate = emoji_messages / total
    exclamation_rate = exclamation_messages / total
    question_rate = question_messages / total
    casual_rate = casual_hits / total
    formal_rate = formal_hits / total

    # Representative samples: mix of short, medium, and long messages.
    sorted_msgs = sorted(messages, key=lambda m: len(str(m.get("text", ""))))
    sample_indices = {
        0,
        total // 4,
        total // 2,
        (3 * total) // 4,
        total - 1,
    }
    evidence_samples = []
    for idx in sorted(sample_indices):
        if 0 <= idx < len(sorted_msgs):
            msg = sorted_msgs[idx]
            evidence_samples.append(
                {
                    "message_id": int(msg["message_id"]),
                    "conversation_id": int(msg["conversation_id"]),
                    "text": str(msg["text"])[:300],
                    "length": len(str(msg["text"])),
                }
            )

    return {
        "avg_message_length": round(avg_length, 2),
        "avg_length_category": categorize_length(avg_length),
        "emoji_usage": emoji_rate >= 0.05,
        "emoji_rate": round(emoji_rate, 4),
        "exclamation_rate": round(exclamation_rate, 4),
        "question_rate": round(question_rate, 4),
        "punctuation_style": categorize_punctuation(exclamation_rate, question_rate),
        "tone": infer_tone(casual_rate, formal_rate),
        "total_messages_analyzed": total,
        "evidence_samples": evidence_samples[:sample_size],
    }


def build_persona(
    messages: Sequence[Dict[str, Any]],
    *,
    target_speaker: str,
    max_evidence_per_item: int,
    min_evidence_count: int,
) -> Dict[str, Any]:
    user_messages = [m for m in messages if str(m.get("speaker", "")).strip() == target_speaker]
    LOGGER.info("Analyzing %s messages for speaker '%s'", len(user_messages), target_speaker)

    habits = extract_with_evidence(
        user_messages,
        HABIT_RULES,
        max_evidence=max_evidence_per_item,
        min_evidence=min_evidence_count,
    )
    personal_facts = extract_with_evidence(
        user_messages,
        PERSONAL_FACT_RULES,
        max_evidence=max_evidence_per_item,
        min_evidence=min_evidence_count,
    )
    personality_traits = extract_with_evidence(
        user_messages,
        PERSONALITY_RULES,
        max_evidence=max_evidence_per_item,
        min_evidence=min_evidence_count,
    )
    communication_style = analyze_communication_style(user_messages)

    return {
        "target_speaker": target_speaker,
        "habits": habits,
        "personal_facts": personal_facts,
        "personality_traits": personality_traits,
        "communication_style": communication_style,
        "metadata": {
            "total_messages_analyzed": len(user_messages),
            "extraction_method": "pattern_matching_with_evidence",
        },
    }


def parse_args() -> PersonaConfig:
    parser = argparse.ArgumentParser(description="Extract user persona from conversation messages.")
    parser.add_argument("--messages-path", type=Path, default=Path("storage/messages.json"))
    parser.add_argument("--output-path", type=Path, default=Path("storage/persona.json"))
    parser.add_argument("--target-speaker", type=str, default="User 1")
    parser.add_argument("--max-evidence-per-item", type=int, default=3)
    parser.add_argument("--min-evidence-count", type=int, default=1)
    parser.add_argument("--limit-messages", type=int, default=None, help="Optional limit for testing.")
    args = parser.parse_args()

    if args.max_evidence_per_item <= 0:
        raise ValueError("--max-evidence-per-item must be > 0")
    if args.min_evidence_count <= 0:
        raise ValueError("--min-evidence-count must be > 0")

    return PersonaConfig(
        messages_path=args.messages_path,
        output_path=args.output_path,
        target_speaker=args.target_speaker,
        max_evidence_per_item=args.max_evidence_per_item,
        min_evidence_count=args.min_evidence_count,
        limit_messages=args.limit_messages,
    )


def main() -> None:
    setup_logging()
    config = parse_args()

    LOGGER.info("Loading messages from %s", config.messages_path)
    messages = load_messages(config.messages_path)
    if config.limit_messages is not None:
        messages = messages[: config.limit_messages]
        LOGGER.info("Limited to %s messages for testing", len(messages))

    persona = build_persona(
        messages,
        target_speaker=config.target_speaker,
        max_evidence_per_item=config.max_evidence_per_item,
        min_evidence_count=config.min_evidence_count,
    )

    save_json(persona, config.output_path)
    LOGGER.info(
        "Saved persona to %s (habits=%s, facts=%s, traits=%s)",
        config.output_path,
        len(persona["habits"]),
        len(persona["personal_facts"]),
        len(persona["personality_traits"]),
    )


if __name__ == "__main__":
    main()

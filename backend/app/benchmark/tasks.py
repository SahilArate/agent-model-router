"""Benchmark tasks with ground-truth answers, generated deterministically from a seed.

Each category is a structured proxy for a capability (extraction, arithmetic reasoning,
code evaluation, counting, tool-call formatting) so correctness can be checked exactly,
without a judge model.
"""

import json
import random
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from app.core.types import RiskLevel, TaskCategory


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class AnswerKind(StrEnum):
    NUMBER = "number"
    WORD = "word"
    JSON = "json"


class Split(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True)
class Task:
    id: str
    category: TaskCategory
    difficulty: Difficulty
    risk: RiskLevel
    kind: AnswerKind
    prompt: str
    expected: str
    wrong_answers: tuple[str, ...]  # well-formed but incorrect, used to simulate plausible errors


# A generator returns (prompt, kind, expected, wrong_answers).
_Generated = tuple[str, AnswerKind, str, tuple[str, ...]]

_RISK_WEIGHTS = ((RiskLevel.LOW, 0.7), (RiskLevel.MEDIUM, 0.2), (RiskLevel.HIGH, 0.1))
_SPLIT_FRACTIONS = (0.4, 0.2)  # train, validation; the remainder is the held-out test split


def _number(value: float) -> str:
    return f"{value:.2f}" if isinstance(value, float) else str(value)


def _extraction(rng: random.Random, difficulty: Difficulty) -> _Generated:
    total = round(rng.uniform(50, 5000), 2)
    extras = {Difficulty.EASY: 0, Difficulty.MEDIUM: 2, Difficulty.HARD: 4}[difficulty]
    labels = ["Subtotal", "Shipping", "Tax", "Deposit paid", "Discount"]
    rng.shuffle(labels)
    distractors = [round(rng.uniform(5, total), 2) for _ in range(extras)]
    lines = [f"{label}: ${value:.2f}" for label, value in zip(labels, distractors, strict=False)]
    lines.insert(rng.randint(0, len(lines)), f"Total due: ${total:.2f}")
    prompt = (
        f"Invoice #{rng.randint(1000, 9999)}\n"
        + "\n".join(lines)
        + "\n\nExtract the total amount due from this invoice. Reply with the number only."
    )
    wrong = tuple(_number(d) for d in distractors) or (_number(round(total + 10, 2)),)
    return prompt, AnswerKind.NUMBER, _number(total), wrong


def _reasoning(rng: random.Random, difficulty: Difficulty) -> _Generated:
    steps = {Difficulty.EASY: 2, Difficulty.MEDIUM: 3, Difficulty.HARD: 5}[difficulty]
    value = rng.randint(5, 40)
    parts = [f"A depot holds {value} units."]
    value_without_last = value
    for i in range(steps):
        before = value
        op = rng.choice(["add", "sub", "mul"])
        if op == "sub" and value <= 6:
            op = "add"
        if op == "add":
            k = rng.randint(2, 30)
            value += k
            parts.append(f"Then it receives {k} more units.")
        elif op == "sub":
            k = rng.randint(1, value - 1)
            value -= k
            parts.append(f"Then it ships out {k} units.")
        else:
            k = rng.randint(2, 4)
            value *= k
            parts.append(f"Then the total is multiplied by {k}.")
        if i < steps - 1:
            value_without_last = value
        else:
            value_without_last = before
    prompt = " ".join(parts) + " How many units are there at the end? Reply with the number only."
    wrong = (str(value_without_last), str(value + 1), str(max(0, value - 1)))
    return prompt, AnswerKind.NUMBER, str(value), tuple(w for w in wrong if w != str(value))


def _coding(rng: random.Random, difficulty: Difficulty) -> _Generated:
    if difficulty is Difficulty.EASY:
        n = rng.randint(5, 400)
        code, value = f"sum(i * i for i in range({n}))", sum(i * i for i in range(n))
    elif difficulty is Difficulty.MEDIUM:
        n, m, r = rng.randint(20, 500), rng.randint(3, 9), rng.randint(0, 2)
        code = f"len([x for x in range({n}) if x % {m} == {r}])"
        value = len([x for x in range(n) if x % m == r])
    else:
        n, a, b = rng.randint(30, 900), rng.randint(3, 6), rng.randint(7, 11)
        code = f"sum(x for x in range({n}) if x % {a} == 0 or x % {b} == 0)"
        value = sum(x for x in range(n) if x % a == 0 or x % b == 0)
    prompt = (
        f"What does this Python expression evaluate to? Reply with the number only.\n\n`{code}`"
    )
    return prompt, AnswerKind.NUMBER, str(value), (str(value + 1), str(max(0, value - 2)))


_WORD_POOL = [
    "river", "stone", "cloud", "ember", "harbor", "meadow", "lantern", "orchard",
    "falcon", "marble", "thistle", "window",
]  # fmt: skip


def _summarization(rng: random.Random, difficulty: Difficulty) -> _Generated:
    gap = {Difficulty.EASY: 3, Difficulty.MEDIUM: 2, Difficulty.HARD: 1}[difficulty]
    words = rng.sample(_WORD_POOL, 5)
    second = rng.randint(3, 6)
    counts = [second + gap, second, max(1, second - 1), 2, 1]
    tokens = [w for w, c in zip(words, counts, strict=True) for _ in range(c)]
    rng.shuffle(tokens)
    prompt = (
        "Which word appears most often in the following text? Reply with that single word.\n\n"
        + " ".join(tokens)
    )
    return prompt, AnswerKind.WORD, words[0], (words[1], words[2])


def _date(rng: random.Random) -> str:
    return f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"


_CITIES = ["Pune", "Oslo", "Lima", "Kyoto", "Cairo", "Lagos", "Quito", "Hanoi", "Turin", "Perth"]
_NAMES = ["asha", "ben", "chen", "dara", "eli", "fatima", "gus", "hana"]
_TOOLS = {
    Difficulty.EASY: [
        ("set_timer", [("minutes", lambda r: r.randint(1, 900))]),
        ("get_weather", [("city", lambda r: r.choice(_CITIES))]),
    ],
    Difficulty.MEDIUM: [
        (
            "book_flight",
            [
                ("origin", lambda r: r.choice(_CITIES)),
                ("destination", lambda r: r.choice(_CITIES)),
                ("date", _date),
            ],
        ),
        (
            "create_event",
            [
                ("title", lambda r: f"sync-{r.randint(1, 99)}"),
                ("date", _date),
                ("duration_minutes", lambda r: r.choice([15, 30, 45, 60])),
            ],
        ),
    ],
    Difficulty.HARD: [
        (
            "create_ticket",
            [
                ("title", lambda r: f"bug-{r.randint(100, 999)}"),
                ("priority", lambda r: r.choice(["low", "high", "urgent"])),
                ("assignee", lambda r: r.choice(_NAMES)),
                ("labels", lambda r: sorted(r.sample(["ui", "api", "db", "infra"], 2))),
                ("due_date", _date),
            ],
        ),
    ],
}


def _tool_use(rng: random.Random, difficulty: Difficulty) -> _Generated:
    name, params = rng.choice(_TOOLS[difficulty])
    args = {key: make(rng) for key, make in params}
    if name == "book_flight" and args["origin"] == args["destination"]:
        args["destination"] = _CITIES[(_CITIES.index(args["origin"]) + 1) % len(_CITIES)]
    signature = f"{name}({', '.join(args)})"
    request = ", ".join(f"{k} = {json.dumps(v)}" for k, v in args.items())
    prompt = (
        f"Available tool: {signature}\n"
        f"User request: call {name} with {request}.\n"
        'Reply with JSON only: {"tool": "<name>", "args": {...}}'
    )

    def encode(a: dict) -> str:
        return json.dumps({"tool": name, "args": a}, sort_keys=True)

    mutated = dict(args)
    key = rng.choice(list(args))
    value = args[key]
    if isinstance(value, int):
        mutated[key] = value + 1
    elif isinstance(value, list):
        mutated[key] = value[:-1]
    else:
        mutated[key] = f"{value}x"
    return prompt, AnswerKind.JSON, encode(args), (encode(mutated),)


_GENERATORS: dict[TaskCategory, Callable[[random.Random, Difficulty], _Generated]] = {
    TaskCategory.EXTRACTION: _extraction,
    TaskCategory.REASONING: _reasoning,
    TaskCategory.CODING: _coding,
    TaskCategory.SUMMARIZATION: _summarization,
    TaskCategory.TOOL_USE: _tool_use,
}


def generate_tasks(n: int, seed: int = 0) -> list[Task]:
    """n tasks, spread evenly over categories and difficulties, with unique prompts."""
    if n < 1:
        raise ValueError("n must be at least 1")
    rng = random.Random(seed)
    categories = list(_GENERATORS)
    difficulties = list(Difficulty)
    risks, weights = zip(*_RISK_WEIGHTS, strict=True)

    tasks: list[Task] = []
    seen: set[str] = set()
    for i in range(n):
        category = categories[i % len(categories)]
        difficulty = difficulties[(i // len(categories)) % len(difficulties)]
        for _ in range(200):
            prompt, kind, expected, wrong = _GENERATORS[category](rng, difficulty)
            if prompt not in seen:
                break
        else:  # pragma: no cover - 200 collisions in a row is effectively impossible
            raise RuntimeError("could not generate a unique task prompt")
        seen.add(prompt)
        tasks.append(
            Task(
                id=f"{category.value}-{i:04d}",
                category=category,
                difficulty=difficulty,
                risk=rng.choices(risks, weights)[0],
                kind=kind,
                prompt=prompt,
                expected=expected,
                wrong_answers=wrong,
            )
        )
    return tasks


def split_tasks(tasks: list[Task], seed: int = 0) -> dict[Split, list[Task]]:
    """40/20/40 train/validation/test split, stratified by category and difficulty."""
    rng = random.Random(seed)
    groups: dict[tuple[TaskCategory, Difficulty], list[Task]] = defaultdict(list)
    for task in tasks:
        groups[(task.category, task.difficulty)].append(task)

    result: dict[Split, list[Task]] = {split: [] for split in Split}
    for key in sorted(groups):
        group = groups[key][:]
        rng.shuffle(group)
        n_train = int(len(group) * _SPLIT_FRACTIONS[0])
        n_val = int(len(group) * _SPLIT_FRACTIONS[1])
        result[Split.TRAIN] += group[:n_train]
        result[Split.VALIDATION] += group[n_train : n_train + n_val]
        result[Split.TEST] += group[n_train + n_val :]
    for split in result:
        result[split].sort(key=lambda t: t.id)
    return result

import json
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any

import chromadb
import gradio as gr
import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR.parent
CHROMA_DIR = BASE_DIR / "chroma_db"
PROMPT_FILE = BASE_DIR / "prompts" / "system_prompt.txt"
COLLECTION_NAME = "nora_clinical_skills"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

load_dotenv(SRC_DIR / ".env")
load_dotenv(SRC_DIR / ".secrets", override=True)

from utils.clients import get_client  # noqa: E402


MODEL = os.getenv("MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CROSSREF_MAILTO = os.getenv("CROSSREF_MAILTO", "").strip()

MAX_HISTORY_MESSAGES = 12
MAX_TOOL_ROUNDS = 4
REQUEST_TIMEOUT_SECONDS = 15

_ai_client: Any | None = None
_collection: Any | None = None
_client_lock = threading.Lock()
_collection_lock = threading.Lock()

SYSTEM_PROMPT = PROMPT_FILE.read_text(encoding="utf-8").strip()


TOOLS = [
    {
        "type": "function",
        "name": "search_nursing_publications",
        "description": (
            "Search Crossref for bibliographic metadata about nursing education, "
            "clinical-skills education, or professional learning publications."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": (
                        "The publication topic, such as simulation-based nursing education."
                    ),
                },
                "maximum_results": {
                    "type": "integer",
                    "description": "Number of results to return, from 1 to 5.",
                },
            },
            "required": ["topic", "maximum_results"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "search_clinical_skills",
        "description": (
            "Search the persistent clinical-skills knowledge base using semantic "
            "similarity. Use for hand hygiene, PPE, sterile technique, SBAR, "
            "documentation, and evidence-based study methods."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The learner's educational clinical-skills question.",
                },
                "topic_filter": {
                    "type": ["string", "null"],
                    "enum": [
                        "hand_hygiene",
                        "ppe",
                        "sterile_technique",
                        "sbar",
                        "documentation",
                        "study_methods",
                        None,
                    ],
                    "description": (
                        "Optional exact topic filter. Use null when the topic is uncertain."
                    ),
                },
            },
            "required": ["query", "topic_filter"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "create_study_plan",
        "description": "Create a structured clinical-skills study schedule.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The educational topic to study.",
                },
                "days": {
                    "type": "integer",
                    "description": "Number of study days, from 1 to 14.",
                },
                "minutes_per_day": {
                    "type": "integer",
                    "description": "Available minutes per day, from 15 to 240.",
                },
                "difficulty": {
                    "type": "string",
                    "enum": ["beginner", "intermediate"],
                    "description": "The learner's desired difficulty.",
                },
            },
            "required": ["topic", "days", "minutes_per_day", "difficulty"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "create_quiz_specification",
        "description": (
            "Create a validated structure for an educational knowledge check. "
            "The model must also retrieve relevant clinical-skills context."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The educational topic for the quiz.",
                },
                "question_count": {
                    "type": "integer",
                    "description": "Number of questions, from 2 to 8.",
                },
                "difficulty": {
                    "type": "string",
                    "enum": ["beginner", "intermediate"],
                },
                "question_type": {
                    "type": "string",
                    "enum": ["multiple_choice", "short_answer", "mixed"],
                },
            },
            "required": [
                "topic",
                "question_count",
                "difficulty",
                "question_type",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


RESTRICTED_PATTERNS = [
    re.compile(r"\bcat(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bdog(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bhoroscope(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bzodiac\b", re.IGNORECASE),
    re.compile(r"\btaylor\s+swift\b", re.IGNORECASE),
]

PROMPT_ATTACK_PATTERNS = [
    re.compile(r"\bsystem\s+prompt\b", re.IGNORECASE),
    re.compile(r"\bdeveloper\s+(?:message|instructions?)\b", re.IGNORECASE),
    re.compile(
        r"\bignore\s+(?:all\s+)?(?:previous|prior)\s+instructions?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:reveal|show|print|quote|repeat|summarize)\b.{0,50}"
        r"\b(?:prompt|hidden instructions?|rules?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:change|modify|replace|override)\b.{0,50}"
        r"\b(?:prompt|instructions?|rules?)\b",
        re.IGNORECASE,
    ),
]

PATIENT_DATA_PATTERNS = [
    re.compile(r"\b(?:patient|client)\s+name\s+is\b", re.IGNORECASE),
    re.compile(r"\bdate\s+of\s+birth\b", re.IGNORECASE),
    re.compile(r"\bhealth\s*card\b", re.IGNORECASE),
    re.compile(r"\bmedical\s+record\s+number\b", re.IGNORECASE),
    re.compile(r"\bMRN\b"),
]

CLINICAL_DECISION_PATTERNS = [
    re.compile(r"\bdiagnos(?:e|is|ing|ed)\b", re.IGNORECASE),
    re.compile(r"\btriag(?:e|ing)\b", re.IGNORECASE),
    re.compile(r"\b(?:dose|dosage|dosing)\b", re.IGNORECASE),
    re.compile(
        r"\bwhat\s+(?:medication|drug|treatment)\s+should\s+(?:i|we)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\binterpret\b.{0,35}\b(?:lab|test|result|scan)\b",
        re.IGNORECASE,
    ),
]


def get_ai_client() -> Any:
    """Create the course OpenAI client only when it is first needed."""
    global _ai_client

    if _ai_client is not None:
        return _ai_client

    with _client_lock:
        if _ai_client is None:
            _ai_client = get_client()

    return _ai_client


def guardrail_response(message: str) -> str | None:
    """Return a refusal or redirection when a request crosses a boundary."""
    if any(pattern.search(message) for pattern in RESTRICTED_PATTERNS):
        return (
            "I cannot help with that restricted topic. I can help with educational "
            "clinical-skills review, study plans, quizzes, or publication metadata."
        )

    if any(pattern.search(message) for pattern in PROMPT_ATTACK_PATTERNS):
        return (
            "I cannot reveal or change hidden instructions. I can explain the "
            "application's safety design at a general level."
        )

    if any(pattern.search(message) for pattern in PATIENT_DATA_PATTERNS):
        return (
            "Please do not enter identifiable patient information. Replace real "
            "details with a fictional, de-identified learning scenario."
        )

    if any(pattern.search(message) for pattern in CLINICAL_DECISION_PATTERNS):
        return (
            "This app is for educational review only. It cannot diagnose, triage, "
            "calculate medication doses, interpret real patient results, or recommend "
            "patient-specific treatment. Ask a general study question using a "
            "fictional scenario instead."
        )

    return None


def get_embedding(text: str) -> list[float]:
    """Create one semantic embedding using the course client pattern."""
    cleaned_text = " ".join(text.replace("\n", " ").split())
    if not cleaned_text:
        raise ValueError("Text for embedding cannot be empty.")

    response = get_ai_client().embeddings.create(
        input=cleaned_text,
        model=EMBEDDING_MODEL,
    )
    return response.data[0].embedding


def get_collection() -> Any:
    """Load the file-persistent Chroma collection once per process."""
    global _collection

    if _collection is not None:
        return _collection

    with _collection_lock:
        if _collection is not None:
            return _collection

        if not CHROMA_DIR.exists():
            raise RuntimeError(
                "The vector database is missing. Run build_vector_db.py first."
            )

        chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            _collection = chroma_client.get_collection(name=COLLECTION_NAME)
        except Exception as exc:
            raise RuntimeError(
                "The Chroma collection is missing. Run build_vector_db.py first."
            ) from exc

    return _collection


def extract_publication_year(item: dict[str, Any]) -> int | None:
    """Extract a publication year from the available Crossref date fields."""
    for field in ("published-print", "published-online", "published", "issued"):
        date_parts = item.get(field, {}).get("date-parts", [])
        if date_parts and date_parts[0]:
            year = date_parts[0][0]
            if isinstance(year, int):
                return year
    return None


def format_authors(authors: list[dict[str, Any]], limit: int = 4) -> list[str]:
    """Convert Crossref author objects into readable names."""
    names: list[str] = []

    for author in authors[:limit]:
        given = str(author.get("given", "")).strip()
        family = str(author.get("family", "")).strip()
        name = " ".join(part for part in (given, family) if part)
        if name:
            names.append(name)

    if len(authors) > limit:
        names.append("et al.")

    return names


def search_nursing_publications(topic: str, maximum_results: int) -> str:
    """Search Crossref and return selected publication metadata as JSON."""
    topic = topic.strip()
    if len(topic) < 3:
        return json.dumps({"error": "Please provide a more specific topic."})

    maximum_results = max(1, min(int(maximum_results), 5))
    params: dict[str, Any] = {
        "query.bibliographic": f"nursing education {topic}",
        "rows": maximum_results,
    }
    if CROSSREF_MAILTO:
        params["mailto"] = CROSSREF_MAILTO

    response = requests.get(
        "https://api.crossref.org/works",
        params=params,
        headers={"User-Agent": "NoraClinicalSkillsCoach/1.0"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    items = response.json().get("message", {}).get("items", [])
    publications = []

    for item in items:
        titles = item.get("title") or []
        containers = item.get("container-title") or []
        publications.append(
            {
                "title": titles[0] if titles else "Untitled",
                "authors": format_authors(item.get("author") or []),
                "year": extract_publication_year(item),
                "journal_or_source": containers[0] if containers else None,
                "doi": item.get("DOI"),
                "url": item.get("URL"),
                "work_type": item.get("type"),
            }
        )

    return json.dumps(
        {
            "topic": topic,
            "result_count": len(publications),
            "publications": publications,
            "notice": (
                "These are bibliographic metadata results. They are not appraised "
                "clinical recommendations or institutional policy."
            ),
        },
        ensure_ascii=False,
    )


def search_clinical_skills(query: str, topic_filter: str | None) -> str:
    """Run semantic search against the persistent clinical-skills collection."""
    query = query.strip()
    if len(query) < 3:
        return json.dumps({"error": "Please provide a more specific study question."})

    collection = get_collection()
    query_embedding = get_embedding(query)

    query_arguments: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": 4,
        "include": ["documents", "metadatas", "distances"],
    }
    if topic_filter:
        query_arguments["where"] = {"topic": topic_filter}

    result = collection.query(**query_arguments)

    matches = []
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    for document, metadata, distance in zip(documents, metadatas, distances):
        matches.append(
            {
                "title": metadata.get("title", "Untitled"),
                "topic": metadata.get("topic", "general"),
                "difficulty": metadata.get("difficulty", "unspecified"),
                "source_type": metadata.get("source_type", "unspecified"),
                "content": document,
                "distance": round(float(distance), 4),
            }
        )

    return json.dumps(
        {
            "query": query,
            "topic_filter": topic_filter,
            "matches": matches,
            "notice": (
                "Use these original educational summaries only. Verify local "
                "institutional policy and instructor guidance."
            ),
        },
        ensure_ascii=False,
    )


def create_study_plan(
    topic: str,
    days: int,
    minutes_per_day: int,
    difficulty: str,
) -> str:
    """Create a deterministic study schedule for the model to present."""
    topic = topic.strip()
    if not topic:
        return json.dumps({"error": "Please provide a study topic."})

    days = max(1, min(int(days), 14))
    minutes_per_day = max(15, min(int(minutes_per_day), 240))
    difficulty = difficulty if difficulty in {"beginner", "intermediate"} else "beginner"

    learn_minutes = max(5, round(minutes_per_day * 0.35))
    recall_minutes = max(5, round(minutes_per_day * 0.30))
    practice_minutes = max(5, round(minutes_per_day * 0.25))
    reflect_minutes = max(
        5,
        minutes_per_day - learn_minutes - recall_minutes - practice_minutes,
    )

    focus_cycle = [
        "define the core purpose and vocabulary",
        "connect principles to a fictional scenario",
        "retrieve the main ideas without notes",
        "compare a strong and weak example",
        "review errors and explain the reasoning aloud",
    ]

    schedule = []
    for day_number in range(1, days + 1):
        schedule.append(
            {
                "day": day_number,
                "focus": focus_cycle[(day_number - 1) % len(focus_cycle)],
                "learn_minutes": learn_minutes,
                "recall_minutes": recall_minutes,
                "practice_minutes": practice_minutes,
                "reflection_minutes": reflect_minutes,
            }
        )

    return json.dumps(
        {
            "topic": topic,
            "difficulty": difficulty,
            "days": days,
            "minutes_per_day": minutes_per_day,
            "schedule": schedule,
            "completion_check": (
                "Explain the topic from memory, complete one fictional practice "
                "scenario, and record remaining questions for an instructor."
            ),
            "safety_note": (
                "This plan supports educational review and does not certify clinical "
                "competency."
            ),
        },
        ensure_ascii=False,
    )


def create_quiz_specification(
    topic: str,
    question_count: int,
    difficulty: str,
    question_type: str,
) -> str:
    """Create a validated quiz structure for grounded question generation."""
    topic = topic.strip()
    if not topic:
        return json.dumps({"error": "Please provide a quiz topic."})

    question_count = max(2, min(int(question_count), 8))
    difficulty = difficulty if difficulty in {"beginner", "intermediate"} else "beginner"
    question_type = (
        question_type
        if question_type in {"multiple_choice", "short_answer", "mixed"}
        else "mixed"
    )

    if question_type == "multiple_choice":
        formats = ["multiple_choice"] * question_count
    elif question_type == "short_answer":
        formats = ["short_answer"] * question_count
    else:
        formats = [
            "multiple_choice" if index % 2 == 0 else "short_answer"
            for index in range(question_count)
        ]

    return json.dumps(
        {
            "topic": topic,
            "question_count": question_count,
            "difficulty": difficulty,
            "question_formats": formats,
            "generation_rules": [
                "Use only retrieved clinical-skills context.",
                "Use fictional educational situations only.",
                "Do not include patient identifiers.",
                "Do not test diagnosis, triage, treatment, or medication dosing.",
                "Do not show the answer key until the learner asks or submits answers.",
            ],
        },
        ensure_ascii=False,
    )


def call_tool(name: str, arguments: dict[str, Any]) -> str:
    """Route a model tool call to the matching Python service."""
    if name == "search_nursing_publications":
        return search_nursing_publications(**arguments)
    if name == "search_clinical_skills":
        return search_clinical_skills(**arguments)
    if name == "create_study_plan":
        return create_study_plan(**arguments)
    if name == "create_quiz_specification":
        return create_quiz_specification(**arguments)
    return json.dumps({"error": f"Unknown tool: {name}"})


def compact_history(history: list[Any]) -> list[dict[str, str]]:
    """Convert recent Gradio history into safe short-term memory."""
    compacted: list[dict[str, str]] = []

    for item in history[-MAX_HISTORY_MESSAGES:]:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")

            if role not in {"user", "assistant"} or not isinstance(content, str):
                continue

            if role == "user" and guardrail_response(content) is not None:
                continue

            compacted.append({"role": role, "content": content})
            continue

        if isinstance(item, (tuple, list)) and len(item) == 2:
            user_text, assistant_text = item
            if isinstance(user_text, str) and guardrail_response(user_text) is None:
                compacted.append({"role": "user", "content": user_text})
            if isinstance(assistant_text, str):
                compacted.append({"role": "assistant", "content": assistant_text})

    return compacted[-MAX_HISTORY_MESSAGES:]


def output_guardrail(text: str) -> str:
    """Block empty output and obvious system-prompt leakage."""
    cleaned = text.strip()
    if not cleaned:
        return "I could not produce a response. Please ask a more specific study question."

    system_prefix = SYSTEM_PROMPT[:100].lower()
    if system_prefix in cleaned.lower():
        return (
            "I cannot reveal hidden instructions. Please ask an educational "
            "clinical-skills question instead."
        )

    return cleaned


def run_agent(message: str, history: list[Any]) -> str:
    """Run the Responses API function-calling loop."""
    refusal = guardrail_response(message)
    if refusal is not None:
        return refusal

    input_list: list[Any] = compact_history(history)
    input_list.append({"role": "user", "content": message})

    try:
        for _ in range(MAX_TOOL_ROUNDS):
            response = get_ai_client().responses.create(
                model=MODEL,
                instructions=SYSTEM_PROMPT,
                tools=TOOLS,
                input=input_list,
                max_output_tokens=900,
            )

            input_list += response.output
            function_calls = [
                item for item in response.output if item.type == "function_call"
            ]

            if not function_calls:
                return output_guardrail(response.output_text or "")

            for function_call in function_calls:
                try:
                    arguments = json.loads(function_call.arguments)
                    tool_result = call_tool(function_call.name, arguments)
                except requests.RequestException as exc:
                    tool_result = json.dumps(
                        {"error": f"The external API request failed: {exc}"}
                    )
                except Exception as exc:
                    tool_result = json.dumps(
                        {"error": f"The tool could not complete the request: {exc}"}
                    )

                input_list.append(
                    {
                        "type": "function_call_output",
                        "call_id": function_call.call_id,
                        "output": tool_result,
                    }
                )

        return "I reached the tool-call limit. Please ask a simpler study question."
    except Exception as exc:
        return (
            "The assistant could not complete the request. Confirm that the course "
            "API credentials, selected Python environment, and vector database are "
            f"available. Details: {exc}"
        )


demo = gr.ChatInterface(
    fn=run_agent,
    title="Nora Clinical Skills Study Coach",
    description=(
        "This is a personal study tool only. Nora searches original clinical-skills summaries, "
        "finds publication metadata, creates study plans, and builds grounded quizzes. "
        "Do not enter any real personal health information."
    ),
    examples=[
        "Explain why gloves do not replace hand hygiene.",
        "Quiz me on SBAR with four beginner questions.",
        "Create a 5-day sterile-technique study plan for 45 minutes per day.",
        "Find three publications about simulation-based nursing education.",
    ],
)


if __name__ == "__main__":
    demo.launch()

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


SAMPLE = ROOT / "data" / "sample_data.xlsx"
os.environ.setdefault("DATA_FILE", str(SAMPLE))
os.environ.pop("OPENAI_API_KEY", None)  # unit tests never call OpenAI


@pytest.fixture(scope="session")
def data():
    from data_loader import load_excel

    return load_excel(str(SAMPLE))


class FakeLLM:
    """Mimics openai.OpenAI().chat.completions.create.

    `extraction` maps a substring of the user message to the JSON the extractor returns.
    The analysis step echoes the data it was given, so tests can check grounding.
    """

    def __init__(self, extraction: dict[str, str]):
        self.extraction = extraction
        self.calls = []
        self.chat = self
        self.completions = self

    def create(self, model, messages, **kwargs):
        self.calls.append(messages)
        system, user = messages[0]["content"], messages[-1]["content"]
        if system.startswith("Eres un extractor"):
            current = user.split("Mensaje:", 1)[-1]  # ignore "Preguntas anteriores"
            content = next((v for k, v in self.extraction.items() if k in current), "{}")
        else:
            content = "**Resultado**: " + user.split("Datos:\n", 1)[-1]
        msg = type("M", (), {"content": content})
        return type("R", (), {"choices": [type("C", (), {"message": msg})]})

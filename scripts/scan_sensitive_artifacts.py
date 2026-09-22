"""Scan the public tree for generic confidentiality and legacy-risk patterns."""
from __future__ import annotations

import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_EXTENSIONS = {".py", ".md", ".txt", ".json", ".csv", ".toml", ".yaml", ".yml", ".gitignore"}
SENSITIVE_EXTENSIONS = {".pkl", ".joblib", ".pt", ".pth", ".ckpt", ".log", ".env"}
SKIP_PARTS = {".git", ".superpowers", ".venv", "__pycache__"}
PLACEHOLDER_VALUES = {"dummy", "example", "example-value", "test", "placeholder", "changeme"}
LEGACY_PATTERNS = (
    r"cageo-d-26-" + r"00782r1",
    r"65\s*/\s*73",
    r"0\.2284",
    r"w/o\s+" + r"physics",
    r"physics\s+(?:" + r"loss|constraint)",
    r"\bno" + r"phys\b",
    r"peak[- ]" + r"aware",
    r"peak\s+(?:" + r"correction|shift|displacement)",
    r"seven-day uranium concentration forecasting in a five-spot " + r"in-situ leaching wellfield",
)


@dataclass(frozen=True)
class Finding:
    path: str
    category: str
    detail: str


def _is_skipped(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    return any(part in SKIP_PARTS for part in relative.parts)


def _read_text(path: Path) -> str | None:
    if path.name == ".gitignore" or path.suffix.lower() in TEXT_EXTENSIONS:
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return None
    return None


def scan_repository(root: Path = ROOT) -> list[Finding]:
    root = Path(root).resolve()
    findings: list[Finding] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and not _is_skipped(item, root)):
        relative = path.relative_to(root).as_posix()
        lower_name = path.name.lower()
        suffix = path.suffix.lower()
        if suffix in SENSITIVE_EXTENSIONS:
            findings.append(Finding(relative, "sensitive extension", suffix))
        if re.search(r"(?:raw|private|protected)[_-]?(?:field|site|well|data|bundle)|row[_-]?predictions", lower_name):
            findings.append(Finding(relative, "protected-style filename", lower_name))

        text = _read_text(path)
        if text is None:
            continue
        for match in re.finditer(
            r"(?im)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|client[_-]?secret)\s*[:=]\s*['\"]?([^\s'\"]+)",
            text,
        ):
            value = match.group(1).strip().lower()
            if value not in PLACEHOLDER_VALUES and not value.startswith("example"):
                findings.append(Finding(relative, "credential pattern", "non-placeholder assignment"))
                break
        if re.search(r"(?i)(?:[a-z]:\\(?:users|documents and settings)\\|/(?:users|home)/[^/\s]+/)", text):
            findings.append(Finding(relative, "absolute local path", "user-directory path"))
        if re.search(r"(?i)\b(?:postgres|mysql|mongodb(?:\+srv)?)://[^\s]+", text):
            findings.append(Finding(relative, "database connection string", "database URL"))
        if re.search(r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b", text):
            findings.append(Finding(relative, "private network address", "private IPv4 address"))

        if suffix == ".csv":
            try:
                header = next(csv.reader(text.splitlines()))
            except (StopIteration, csv.Error):
                header = []
            normalized = {item.strip().lower() for item in header}
            if normalized & {"lat", "lon", "latitude", "longitude", "coordinates", "site_coordinates"}:
                findings.append(Finding(relative, "coordinate-like columns", "coordinate header"))

        parts = path.relative_to(root).parts
        active_scientific_file = not (len(parts) >= 2 and parts[0] == "docs" and parts[1] == "superpowers")
        if active_scientific_file:
            lowered = text.lower()
            for pattern in LEGACY_PATTERNS:
                if re.search(pattern, lowered):
                    findings.append(Finding(relative, "legacy scientific terminology", pattern))
                    break
    return findings


def main() -> int:
    findings = scan_repository()
    if findings:
        print("SENSITIVE ARTIFACT SCAN: FAIL")
        for finding in findings:
            print(f"- {finding.path}: {finding.category} ({finding.detail})")
        return 1
    print("SENSITIVE ARTIFACT SCAN: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

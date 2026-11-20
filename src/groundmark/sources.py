import hashlib
import json
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

TRACKING_KEYS = {"fbclid", "gclid", "msclkid"}


def canonical_url(value: str) -> str | None:
    try:
        parts = urlsplit(value.strip())
        if (
            parts.scheme not in {"http", "https"}
            or not parts.hostname
            or parts.username is not None
            or parts.password is not None
            or any(char.isspace() for char in value.strip())
        ):
            return None
        host = parts.hostname.lower().rstrip(".")
        if ":" in host:
            host = f"[{host}]"
        default = 443 if parts.scheme == "https" else 80
        port = f":{parts.port}" if parts.port and parts.port != default else ""
        query = urlencode(
            sorted(
                (key, val)
                for key, val in parse_qsl(parts.query, keep_blank_values=True)
                if key.lower() not in TRACKING_KEYS and not key.lower().startswith("utm_")
            )
        )
        return urlunsplit((parts.scheme, host + port, parts.path or "/", query, ""))
    except (ValueError, UnicodeError):
        return None


def text_digest(text: str) -> str:
    return hashlib.sha256(" ".join(text.casefold().split()).encode()).hexdigest()


def clean_text(text: str) -> str:
    return re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)


def untrusted_source(source_id: str, **data: str) -> str:
    payload = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e")
    return f'<untrusted_source id="{source_id}">{payload}</untrusted_source>'


def private_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as temp:
        temporary = Path(temp.name)
        try:
            json.dump(value, temp, ensure_ascii=False)
            temp.flush()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

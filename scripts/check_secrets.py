import argparse
import re
import subprocess
from pathlib import Path

from dotenv import dotenv_values


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args])


LITERAL_CREDENTIAL = re.compile(
    rb"(?mi)^[+-]?[ \t]*(?:export[ \t]+|-[ \t]+)?[\"']?"
    rb"(?:NEBIUS_API_KEY|TAVILY_API_KEY|POSTGRES_PASSWORD|GF_SECURITY_ADMIN_PASSWORD)"
    rb"[\"']?[ \t]*[:=][ \t]*[\"']?([A-Za-z0-9_./+=-]{8,})[\"']?[ \t\r]*$"
)
KNOWN_TOKEN = re.compile(
    rb"\b(?:tvly-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{28,}|gh[pousr]_[A-Za-z0-9]{30,}"
    rb"|github_pat_[A-Za-z0-9_]{50,}|sk-(?:proj-)?[A-Za-z0-9_-]{28,}"
    rb"|v1\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{24,})\b"
    rb"|-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----"
)


def contains_credential(payload: bytes) -> bool:
    return KNOWN_TOKEN.search(payload) is not None or any(
        not match.group(1).lower().startswith((b"example", b"replace", b"your_", b"changeme"))
        for match in LITERAL_CREDENTIAL.finditer(payload)
    )


def tracked_only() -> int:
    paths = git("ls-tree", "-rz", "--name-only", "HEAD").split(b"\0")
    tracked = [path.decode("utf-8", errors="surrogateescape") for path in paths if path]
    findings = sum(contains_credential(git("show", f"HEAD:{path}")) for path in tracked)
    diff_available = all(
        subprocess.run(
            ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"], check=False
        ).returncode
        == 0
        for branch in ("main", "develop")
    )
    diff_finding = False
    print(f"Credential pattern scan: {len(tracked)} HEAD blobs")
    if diff_available:
        diff_finding = contains_credential(
            git("diff", "--no-ext-diff", "--binary", "main..develop")
        )
        print("Credential pattern scan: main..develop Git patch")
    else:
        print("main..develop patch skipped (local branch ref absent)")
    print("Exact local credential comparison skipped (local values deliberately unread)")
    if findings or diff_finding:
        print(f"Credential patterns found: {findings} HEAD blobs, diff={diff_finding}")
        return 1
    print("Credential pattern scan passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracked-only", action="store_true")
    if parser.parse_args().tracked_only:
        return tracked_only()
    secrets = []
    for filename, keys in (
        (".env", ("NEBIUS_API_KEY", "TAVILY_API_KEY")),
        (".env.compose", ("POSTGRES_PASSWORD", "GF_SECURITY_ADMIN_PASSWORD")),
    ):
        if Path(filename).is_file():
            values = dotenv_values(filename, interpolate=False)
            secrets.extend(values[key].encode() for key in keys if values.get(key))
    if not secrets:
        print("Credential check skipped (local credentials absent)")
        return 0
    paths = git("ls-files", "-z").split(b"\0")
    tracked = [Path(path.decode("utf-8", errors="surrogateescape")) for path in paths if path]
    for path in tracked:
        if path.is_file() and any(secret in path.read_bytes() for secret in secrets):
            print("Credential check failed in a tracked file")
            return 1
    for args in (("diff", "--cached", "--binary"), ("diff", "57c5412", "--binary", "HEAD")):
        payload = git(*args)
        if any(secret in payload for secret in secrets):
            print("Credential check failed in a Git diff")
            return 1
    print("Credential check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

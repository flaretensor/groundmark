import subprocess
from pathlib import Path

from dotenv import dotenv_values


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args])


def main() -> int:
    if not Path(".env").is_file():
        print("Credential check skipped (.env absent)")
        return 0
    values = dotenv_values(".env", interpolate=False)
    secrets = [
        values[key].encode() for key in ("NEBIUS_API_KEY", "TAVILY_API_KEY") if values.get(key)
    ]
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

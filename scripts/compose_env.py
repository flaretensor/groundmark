import os
import secrets
from pathlib import Path
from stat import S_IMODE
from urllib.parse import quote


def main() -> None:
    path = Path(".env.compose")
    if path.is_symlink():
        raise SystemExit(".env.compose must be a regular file")
    if path.exists():
        if not path.is_file() or S_IMODE(path.stat().st_mode) != 0o600:
            raise SystemExit(".env.compose must be a regular file with mode 0600")
        return
    password = secrets.token_urlsafe(32)
    db = f"groundmark:{quote(password, safe='')}@postgres:5432"
    env = {
        "POSTGRES_PASSWORD": password,
        "GROUNDMARK_CHECKPOINT_URL": f"postgresql://{db}/groundmark",
        "MLFLOW_BACKEND_STORE_URI": f"postgresql+psycopg2://{db}/mlflow",
        "GF_SECURITY_ADMIN_PASSWORD": secrets.token_urlsafe(32),
    }
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as file:
        for key, val in env.items():
            print(f"{key}={val}", file=file)


if __name__ == "__main__":
    main()

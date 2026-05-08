"""
Chamar a mesma cadeia que `./garage_alpr.sh` a partir de Python, sem duplicar regras.

Usa `garage_alpr.sh --json` no repositório (enhance + ALPR + códigos de saída estáveis).

Para reconhecimento *só* OpenALPR (sem enhance do wrapper), compile o binding e use
`openalpr.openalpr.Alpr` com o `openalpr.conf` adequado — comportamento diferente.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, MutableMapping


class GarageAlprError(RuntimeError):
    def __init__(self, message: str, *, exit_code: int, payload: Mapping[str, Any]):
        super().__init__(message)
        self.exit_code = exit_code
        self.payload = payload


def _repo_root() -> Path:
    env = os.environ.get("OPENALPR_REPO")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parent.parent


def read_plate(
    image_path: str | Path,
    *,
    repo_root: Path | None = None,
    env: MutableMapping[str, str] | None = None,
) -> dict[str, Any]:
    """
    Executa `garage_alpr.sh --json` e devolve o dict JSON de sucesso.

    Levanta `GarageAlprError` com `exit_code` e `payload` (JSON do stderr) em falha.
    """
    root = repo_root or _repo_root()
    script = root / "garage_alpr.sh"
    if not script.is_file():
        raise FileNotFoundError(f"garage_alpr.sh not found under repo root: {root}")

    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    p = subprocess.run(
        [str(script), "--json", str(Path(image_path).expanduser())],
        cwd=str(root),
        env=run_env,
        capture_output=True,
        text=True,
    )

    if p.returncode == 0:
        line = (p.stdout or "").strip()
        if not line:
            raise GarageAlprError("empty stdout", exit_code=0, payload={})
        return json.loads(line)

    err_line = (p.stderr or "").strip()
    try:
        payload = json.loads(err_line) if err_line else {}
    except json.JSONDecodeError:
        payload = {"error": err_line or "unknown", "raw_stderr": err_line}
    msg = str(payload.get("error", "garage_alpr_failed"))
    raise GarageAlprError(msg, exit_code=p.returncode, payload=payload)


def main() -> None:
    import sys

    if len(sys.argv) != 2:
        print("Usage: python3 -m garage_alpr_client <image>", file=sys.stderr)
        sys.exit(2)
    try:
        d = read_plate(sys.argv[1])
    except GarageAlprError as e:
        print(json.dumps(e.payload), file=sys.stderr)
        sys.exit(e.exit_code)
    except (OSError, json.JSONDecodeError) as e:
        print(json.dumps({"error": str(e), "code": 2}), file=sys.stderr)
        sys.exit(2)
    print(json.dumps(d))


if __name__ == "__main__":
    main()

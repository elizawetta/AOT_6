import contextlib
import importlib.metadata
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.llm import MLXClient


def emit(value):
    sys.stdout.write(json.dumps(value, ensure_ascii=False)+'\n')
    sys.stdout.flush()


try:
    config = json.loads(Path(sys.argv[1]).read_text())
    with contextlib.redirect_stdout(sys.stderr):
        client = MLXClient(config, ROOT)
    emit({'ready': True, 'packages': {p:importlib.metadata.version(p) for p in ['mlx-lm','mlx','transformers']}, 'python':sys.executable})
    for line in sys.stdin:
        request = json.loads(line)
        try:
            with contextlib.redirect_stdout(sys.stderr):
                value = client.generate(request['system'], request['user'])
            emit({'generation':asdict(value)})
        except Exception as exc:
            emit({'error':f'{type(exc).__name__}: {exc}'})
except Exception as exc:
    emit({'error':f'{type(exc).__name__}: {exc}'})
    sys.exit(1)

import json
import time
import subprocess
import select
import sys
from dataclasses import dataclass
from pathlib import Path


class LLMUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class Generation:
    raw_text: str
    input_tokens: int
    output_tokens: int
    elapsed_seconds: float
    token_measurement: str = 'backend_reported'


class MLXClient:
    def __init__(self, config: dict, project_root: Path):
        try:
            import mlx.core as mx
            from mlx_lm import load
            model_path = (project_root / config['model_path']).resolve()
            if not model_path.is_dir():
                raise FileNotFoundError(f'Local model does not exist: {model_path}')
            if model_path.name != config['revision']:
                raise ValueError('Local snapshot directory must match the pinned revision')
            capacity = json.loads((model_path / 'config.json').read_text())['max_position_embeddings']
            if config['max_context_tokens'] > capacity:
                raise ValueError('Configured context capacity exceeds model metadata')
            self.model, self.tokenizer = load(str(model_path))
            self.mx = mx
            self.config = config
        except Exception as exc:
            raise LLMUnavailable(f'MLX model could not be loaded: {type(exc).__name__}: {exc}') from exc

    def generate(self, system: str, user: str) -> Generation:
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_tokens = len(self.tokenizer.encode(prompt))
        if input_tokens + self.config['max_output_tokens'] > self.config['max_context_tokens']:
            raise ValueError('Context would exceed model capacity; silent truncation is forbidden')
        self.mx.random.seed(self.config['seed'])
        start = time.perf_counter()
        text = []
        output_tokens = 0
        for response in stream_generate(self.model, self.tokenizer, prompt=prompt, max_tokens=self.config['max_output_tokens'], sampler=make_sampler(temp=self.config['temperature'])):
            text.append(response.text)
            output_tokens = response.generation_tokens
            if time.perf_counter() - start > self.config['timeout_seconds']:
                raise LLMUnavailable('Generation time limit exceeded')
        return Generation(''.join(text), input_tokens, output_tokens, time.perf_counter() - start)


class MLXProcessClient:
    def __init__(self, config_path: Path, config: dict, project_root: Path):
        self.timeout = config['timeout_seconds']
        executable = (project_root / config['python_executable']).absolute()
        if not executable.is_file():
            raise LLMUnavailable(f'Model interpreter does not exist: {executable}')
        self.process = subprocess.Popen([str(executable), str(project_root/'scripts/mlx_worker.py'), str(config_path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.backend_info = self._receive()
        if not self.backend_info.get('ready'):
            self.close()
            raise LLMUnavailable(str(self.backend_info))

    def _receive(self):
        ready, _, _ = select.select([self.process.stdout], [], [], self.timeout)
        if not ready:
            self.close()
            raise LLMUnavailable('Local model worker timed out')
        line = self.process.stdout.readline()
        if not line:
            raise LLMUnavailable('Local model worker exited')
        return json.loads(line)

    def generate(self, system: str, user: str) -> Generation:
        if self.process.poll() is not None:
            raise LLMUnavailable('Local model worker is not running')
        self.process.stdin.write(json.dumps({'system':system,'user':user},ensure_ascii=False)+'\n')
        self.process.stdin.flush()
        value = self._receive()
        if 'error' in value:
            raise LLMUnavailable(value['error'])
        return Generation(**value['generation'])

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)


def create_client(config_path: str | Path, project_root: str | Path):
    config = json.loads(Path(config_path).read_text())
    if config.get('backend') != 'mlx':
        raise ValueError('Only the configured local MLX backend is implemented')
    for key in ['model', 'revision', 'model_path', 'temperature', 'seed', 'max_output_tokens', 'max_context_tokens', 'timeout_seconds']:
        if key not in config:
            raise ValueError(f'Missing model configuration field: {key}')
    if config['temperature'] < 0 or config['max_output_tokens'] <= 0 or config['timeout_seconds'] <= 0:
        raise ValueError('Invalid generation parameters')
    if config.get('python_executable'):
        return MLXProcessClient(Path(config_path).resolve(), config, Path(project_root))
    return MLXClient(config, Path(project_root))

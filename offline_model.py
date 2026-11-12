"""Run the paired pilot through llama.cpp without opening a network socket."""

import json
import re
import subprocess
import tempfile
from pathlib import Path


class OfflineModel:
    def __init__(self, executable, weights, projector, ffmpeg_dir):
        self.name = weights.stem
        self.command = [
            str(executable.resolve(strict=True)), "-m", str(weights.resolve(strict=True)),
            "--mmproj", str(projector.resolve(strict=True)), "-ngl", "0",
            "--no-mmproj-offload", "-c", "4096", "--temp", "0", "-n", "48",
            "--jinja", "--video-fps", "2",
            "--video-ffmpeg-dir", str(ffmpeg_dir.resolve(strict=True)),
        ]

    def __call__(self, prompt, labels, *, video=None, wav=None):
        prompt = prompt.replace("\n", " ")
        grammar = "root ::= " + json.dumps('{"label":"') + " ("
        grammar += " | ".join(map(json.dumps, labels)) + ") " + json.dumps('"}') + "\n"
        with tempfile.TemporaryDirectory() as temporary:
            grammar_file = Path(temporary) / "label.gbnf"
            grammar_file.write_text(grammar, encoding="ascii")
            command = self.command + ["--grammar-file", str(grammar_file)]
            if video is None:
                stdin = (prompt + "\n/exit\n").encode("utf-8")
            else:
                command += ["--video", str(video), "--audio", str(wav), "-p", prompt]
                stdin = b""
            try:
                result = subprocess.run(command, input=stdin, capture_output=True, timeout=240)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError("Offline model timed out after 240 seconds.") from error
        response = result.stdout.decode("utf-8", errors="replace")
        if result.returncode:
            diagnostic = result.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"Offline model exited {result.returncode}: {diagnostic[-1200:]}")
        matches = re.findall(r'(?m)^\s*(\{"label":"[a-z_]+"\})\s*$', response)
        raw = matches[0] if len(matches) == 1 else response.strip()
        prediction = json.loads(raw)["label"] if len(matches) == 1 else None
        return {"prediction": prediction if prediction in labels else None, "response": raw,
                "model": self.name, "usage": None}

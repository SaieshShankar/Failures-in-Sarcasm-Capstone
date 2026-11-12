"""Check both local model input paths on a short unlabeled clip."""

import argparse
import json
from pathlib import Path

from grounded import resolve_openface, write_cues
from pilot import predict, prepare_inputs, server_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("media", type=Path)
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--openface", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    media = args.media.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    wav, video = prepare_inputs(media, output)
    cues = write_cues(wav, video, resolve_openface(args.openface), output)
    model = server_json(args.port, "/v1/models")["data"][0]["id"]
    results = {
        "kind": "unlabeled_model_smoke",
        "IS": predict(args.port, model, args.transcript, video=video, wav=wav),
        "GS": predict(args.port, model, args.transcript, cues=cues),
    }
    report = json.dumps(results, indent=2) + "\n"
    (output / "model-smoke.json").write_text(report, encoding="utf-8")
    print(report, end="")
    if any(results[key]["prediction"] is None for key in ("IS", "GS")):
        raise SystemExit("A model response lacked a valid label.")


if __name__ == "__main__":
    main()

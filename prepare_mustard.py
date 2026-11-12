"""Join verified MUStARD++R annotation IDs to local target-utterance video files."""

import argparse
import json
import random
from pathlib import Path


def prepare(annotations, media_dir, count, seed):
    records = json.loads(annotations.read_text(encoding="utf-8-sig"))
    if not isinstance(records, dict):
        raise ValueError("Expected an object mapping clip IDs to annotations. See README.")
    by_label = {False: [], True: []}
    missing = []
    for clip_id, record in sorted(records.items()):
        if not isinstance(record, dict) or type(record.get("sarcasm")) is not bool:
            raise ValueError(f"{clip_id}: expected a Boolean sarcasm label")
        transcript = record.get("utterance")
        if not isinstance(transcript, str) or not transcript.strip():
            raise ValueError(f"{clip_id}: expected a nonempty target utterance")
        if not clip_id or Path(clip_id).name != clip_id or not all(
            character.isascii() and (character.isalnum() or character in "_-") for character in clip_id
        ):
            raise ValueError(f"Unsafe clip ID: {clip_id!r}")
        media = media_dir / f"{clip_id}.mp4"
        if not media.is_file() or not media.stat().st_size:
            missing.append(clip_id)
            continue
        by_label[record["sarcasm"]].append({
            "id": clip_id, "media": str(media.resolve()), "transcript": transcript.strip(),
            "label": "sarcastic" if record["sarcasm"] else "not_sarcastic",
        })
    if min(map(len, by_label.values())) < count // 2:
        raise ValueError(f"Need {count // 2} local clips per class. Found "
                         f"{len(by_label[True])} sarcastic and {len(by_label[False])} non-sarcastic.")
    rng = random.Random(seed)
    selected = rng.sample(by_label[True], count // 2) + rng.sample(by_label[False], count // 2)
    rng.shuffle(selected)
    return {"dataset": "MUStARD++R", "seed": seed, "missing_media_count": len(missing), "clips": selected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("media_dir", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--count", type=int, choices=range(2, 21, 2), default=12)
    parser.add_argument("--seed", type=int, default=401)
    args = parser.parse_args()
    result = prepare(args.annotations, args.media_dir, args.count, args.seed)
    with args.manifest.open("x", encoding="utf-8") as destination:
        json.dump(result, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
    print(f"Selected {len(result['clips'])} clips. Missing local media for {result['missing_media_count']} IDs.")


if __name__ == "__main__":
    main()

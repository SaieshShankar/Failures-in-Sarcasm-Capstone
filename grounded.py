"""Measure acoustic and facial behavior in one utterance clip."""

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sysconfig
from pathlib import Path
from statistics import median

# Managed Windows shells can omit the value used by native audio wheels.
if sysconfig.get_platform() == "win-amd64":
    os.environ.setdefault("PROCESSOR_ARCHITECTURE", "AMD64")


def ffmpeg_path():
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def run(command, log, cwd=None):
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, timeout=180)
    except subprocess.TimeoutExpired as error:
        log.write_bytes((error.stdout or b"") + b"\n" + (error.stderr or b""))
        raise RuntimeError(f"{Path(command[0]).name} timed out. See {log}") from error
    log.write_bytes(result.stdout + b"\n" + result.stderr)
    if result.returncode:
        raise RuntimeError(f"{Path(command[0]).name} exited {result.returncode}. See {log}")


def extract_waveform(media, output):
    wav = output / "speech.wav"
    run([ffmpeg_path(), "-nostdin", "-y", "-i", str(media), "-map", "0:a:0",
         "-ac", "1", "-ar", "16000", str(wav)], output / "audio.log")
    return wav


def measure_audio(wav, output):
    import opensmile

    config = {"feature_set": opensmile.FeatureSet.eGeMAPSv02}
    frames = opensmile.Smile(feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
                            **config).process_file(wav)
    stats = opensmile.Smile(feature_level=opensmile.FeatureLevel.Functionals,
                           **config).process_file(wav)
    frames.reset_index().drop(columns="file").to_csv(output / "audio_frames.csv", index=False)
    stats.reset_index().drop(columns="file").to_csv(output / "audio_functionals.csv", index=False)
    pitch = frames["F0semitoneFrom27.5Hz_sma3nz"].dropna()
    voiced = pitch[pitch > 0]
    loudness = frames["Loudness_sma3"].dropna()
    return {
        "frames": len(frames),
        "functional_columns": len(stats.columns),
        "voiced_frames": len(voiced),
        "median_f0_hz": round(float(27.5 * 2 ** (voiced.median() / 12)), 1) if len(voiced) else None,
        "f0_p10_to_p90_semitones": round(float(voiced.quantile(.9) - voiced.quantile(.1)), 2)
        if len(voiced) else None,
        "median_opensmile_loudness": round(float(loudness.median()), 3) if len(loudness) else None,
    }


def resolve_openface(executable):
    located = shutil.which(str(executable)) or str(executable)
    path = Path(located).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"OpenFace FeatureExtraction executable unavailable: {path}")
    return path


def measure_video(media, executable, output, confidence):
    landmark_model = executable.parent / "model" / "main_clnf_general.txt"
    if not landmark_model.is_file():
        raise FileNotFoundError(f"Missing bundled OpenFace CLNF model: {landmark_model}")
    run([str(executable), "-f", str(media), "-out_dir", str(output), "-of", "face",
         "-mloc", str(landmark_model), "-aus", "-pose", "-gaze", "-quiet"],
        output / "openface.log", executable.parent)
    with (output / "face.csv").open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source, skipinitialspace=True)
        columns = {key.strip() for key in reader.fieldnames or ()}
        required = {"timestamp", "success", "confidence", "AU12_r", "AU04_r",
                    "gaze_angle_x", "gaze_angle_y", "pose_Rx", "pose_Ry"}
        if missing := required - columns:
            raise ValueError(f"OpenFace output missing columns: {', '.join(sorted(missing))}")
        frames = [{key.strip(): value for key, value in row.items()} for row in reader]
    good = [row for row in frames
            if int(row["success"]) == 1 and float(row["confidence"]) >= confidence]

    def summarize(column, degrees=False):
        values = [float(row[column]) for row in good if math.isfinite(float(row[column]))]
        if values:
            value = median(values)
            return round(math.degrees(value) if degrees else value, 2)
        return None

    return {
        "frames": len(frames), "usable_frames": len(good), "min_confidence": confidence,
        "landmark_model": "CLNF general",
        "coverage": round(len(good) / len(frames), 3) if frames else 0,
        "track_scope": "automatically selected face; speaker attribution unverified",
        "median_au12_lip_corner_pull_intensity": summarize("AU12_r"),
        "median_au04_brow_lower_intensity": summarize("AU04_r"),
        "median_gaze_angle_deg": [summarize(key, True) for key in ("gaze_angle_x", "gaze_angle_y")],
        "median_head_pitch_yaw_deg": [summarize(key, True) for key in ("pose_Rx", "pose_Ry")],
    }


def extract(media, executable, output, confidence=.8):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    media = media.resolve(strict=True)
    wav = extract_waveform(media, output)
    return write_cues(wav, media, executable, output, confidence)


def write_cues(wav, video_media, executable, output, confidence=.8):
    cues = {"audio": measure_audio(wav, output),
            "video": measure_video(video_media, executable, output, confidence)}
    (output / "cues.json").write_text(json.dumps(cues, indent=2) + "\n", encoding="utf-8")
    return cues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("media", type=Path, help="Clip with audio and video streams")
    parser.add_argument("--openface", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cues = extract(args.media, resolve_openface(args.openface), args.output)
    print(json.dumps(cues, indent=2))
    if not cues["audio"]["voiced_frames"] or not cues["video"]["usable_frames"]:
        raise SystemExit("Smoke check requires both voiced audio and usable face frames.")


if __name__ == "__main__":
    main()

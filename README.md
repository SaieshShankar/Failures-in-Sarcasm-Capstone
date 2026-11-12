# Failures-in-Sarcasm-Capstone

Capstone Project 2028. This milestone implements grounded audio/video extraction and a paired single-pass sarcasm pilot.

| Condition | Inputs to the same local model |
| --- | --- |
| IS | Target transcript, decoded waveform, and the original video stream |
| GS | Target transcript and measured openSMILE/OpenFace observations |

The runner writes the reference label and both predictions side by side. It computes accuracy, macro-F1, and confusion matrices on the common set of clips with two valid predictions. The model prompt contains the target transcript and condition-specific evidence only.

## Windows setup and extractor smoke check

Use 64-bit Python 3.11 and the Microsoft x64 Visual C++ redistributable required by [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Windows-Installation). From the repository directory in PowerShell:

```powershell
& .\setup_windows.ps1
$capPython = '.\.venv\Scripts\python.exe'
$capOpenFace = '.\.local\openface\OpenFace_2.2.0_win_x64\FeatureExtraction.exe'
curl.exe -fL -o .local\default.wmv https://raw.githubusercontent.com/TadasBaltrusaitis/OpenFace/master/samples/default.wmv
& $capPython grounded.py .local\default.wmv --openface $capOpenFace --output runs\extractor-smoke
```

The smoke command writes timestamped eGeMAPS v02 descriptors, 88 clip-level audio functionals, a frame-level OpenFace CSV, and `cues.json`. It exits unsuccessfully if either voiced audio frames or usable face frames are absent. OpenFace uses its bundled **CLNF general** landmark tracker. The confidence cutoff is 0.8. The automatically selected face is unverified as the speaker, and gaze, head pose, and action-unit intensities are observations. Output directories must be new so each run has its own evidence.

See [Windows smoke results](experiments/windows-smoke.json) for the verified extractor and local model outputs. The MUStARD++R comparison is pending access to the official clips and labels.

## MUStARD++R input

Use the [authors' MCIB release instructions](https://github.com/sljgkjhwe/MCIB#quick-start) for the Quark share and extraction code. Obtain the **R** edition's cleaned target-utterance media and corresponding gold labels. Inspect the release before normalization. Keep these files under `data/`, which is ignored by Git. Check that the clip plays, includes an audio stream, and corresponds to the text before sampling. The annotation export schema below is the runner's normalized input. If the release uses a different schema, convert from the official IDs, target utterances, and gold labels. Name each local MP4 after its official clip ID.

```json
{
  "clip_001": {"utterance": "Example transcript", "sarcasm": true},
  "clip_002": {"utterance": "Another example", "sarcasm": false}
}
```

```powershell
& $capPython prepare_mustard.py data\mustard-r-annotations.json data\mustard-r-clips data\pilot.json --count 12 --seed 401
```

## Local model and paired run

Use [Gemma 4 E2B IT GGUF](https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF) with a matching multimodal projector and a [llama.cpp Windows x64 release](https://github.com/ggml-org/llama.cpp/releases) that supports `input_video` and `input_audio`. The server needs a directory containing both `ffmpeg.exe` and `ffprobe.exe`, for example from a [Gyan FFmpeg essentials build](https://github.com/GyanD/codexffmpeg/releases). Gemma encodes video as frames and receives the waveform explicitly. Its E2B variant accepts up to 30 seconds of audio. Use short target utterance clips. This runs on CPU, listens on loopback, and requires no API token.

In one terminal, substituting your executable and model paths:

```powershell
& .\.local\llama\llama-server.exe -m .local\gemma-4-E2B-it-Q4_0.gguf --mmproj .local\mmproj-gemma-4-E2B-it-Q8_0.gguf --alias gemma-4-e2b-it --host 127.0.0.1 --port 8080 -ngl 0 --no-mmproj-offload -c 4096 --parallel 1 --video-ffmpeg-dir .local\ffmpeg\bin --video-fps 2
```

In a second terminal from the same directory:

```powershell
& .\.venv\Scripts\python.exe pilot.py data\pilot.json --openface .local\openface\OpenFace_2.2.0_win_x64\FeatureExtraction.exe --output runs\mustard-r-pilot
```

Inspect `predictions.jsonl`, `comparison.csv`, `summary.json`, and each clip's `cues.json`. Record model files and runtime version alongside any shared results. The pilot measures performance on a small convenience sample from accessible official clips. A disagreement between IS and GS alone does not identify a perception or reasoning failure.

To verify both request paths on an unlabeled short MP4 with audio, use `smoke_model.py` with the clip's transcript:

```powershell
& $capPython smoke_model.py path\to\clip.mp4 --transcript 'Transcript of the clip.' --openface $capOpenFace --output runs\model-smoke
```

## Implementation references

- [TextMI (2023)](https://arxiv.org/abs/2303.15430) motivates expressing acoustic and visual observations as text.
- [When Models Hear What They Expect (2026)](https://aclanthology.org/2026.findings-emnlp.1464/) motivates describing pitch and related measurements without encoding them as a sarcasm decision rule.
- [Conditional Information Bottleneck for Multimodal Fusion (AAAI 2026)](https://ojs.aaai.org/index.php/AAAI/article/view/39874) defines the MUStARD++R reconstruction and release.

# Instructions for Claude Code (TomatoLocalization)

## Persona & User Verification
- Always address the user by his name: **Mihai**. (This acts as an indicator that you are correctly following system context).

## Core Directives & Token Economy
- Be extremely concise. Output minimal code diffs instead of rewriting full unchanged files.
- Do NOT explain basic Python/PyTorch concepts in text.
- Refer to `AGENTS.md` for architecture, inter-agent communication, and payload schemas (`LocalizedTomato`, `DetectedInstance`).
- NEVER delete, overwrite, or re-generate `AGENTS.md`.

## System Invariants
- Pipeline operates strictly on Rectified Left frames (`stereo-original-image/left/`).
- Spatial coordinates and dimensions are strictly in millimeters (`mm`).
- Point cloud processing and sphere fitting in `pipeline.py` MUST be wrapped in `try/except` with `continue`.
- Preserve `flush=True` on all `print()` calls for real-time stdout logging.
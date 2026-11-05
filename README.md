# Multimodal Smart Adaptive Learning System

Research project for multimodal learner engagement estimation and adaptive learning. This repository contains the React/Vite dashboard, FastAPI webcam backend, and existing research code.

## Application structure

- `frontend/` — React, TypeScript, and Vite dashboard with dashboard, learning, analytics, settings, and live webcam pages.
- `Pupil-dilation/` — FastAPI eye tracking, screen-gaze estimation, blink detection, I3D feature extraction, and W0-P inference.
- `docs/live-engagement-setup.md` — setup, external model assets, environment variables, and validation details.
- `cmose-few-shot-personalisation/`, `eye-stream-on-daisee/`, `personalisation/`, `models/`, and `Fixation-Duration_Saccade-Velocity/` — existing research experiments.

## Quick start

1. Install Python dependencies from `Pupil-dilation/requirements.txt` and Node dependencies from `frontend/package-lock.json`.
2. Set `I3D_LIBRARY_DIR` to the external `video_features` library and `W0P_CHECKPOINT_DIR` to the exported checkpoint directory. The export directory should contain `w0p_checkpoints/` and `w0_base_checkpoints/`.
3. Set `frontend/.env.local` with `VITE_API_BASE_URL=http://127.0.0.1:8001` (this is also the default).
4. Start the backend from `Pupil-dilation/` with `python -m uvicorn api:app --host 127.0.0.1 --port 8001`.
5. Start the frontend from `frontend/` with `npm ci && npm run dev`.

The browser requests webcam permission and sends frames to `/analyze-frame`. The backend returns eye metrics and, after collecting a clip, real I3D/W0-P predictions. Model weights and the upstream `video_features` library are external assets and are not stored in this repository. See the setup guide for complete commands and requirements.

## Existing research dependencies

The live application dependencies and startup steps are documented above. Other experiment folders may have their own data, model, and environment requirements; consult their accompanying notebooks and documentation before running those experiments.

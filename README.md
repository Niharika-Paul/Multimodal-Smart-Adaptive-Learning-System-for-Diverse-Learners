# Multimodal Smart Adaptive Learning System for Diverse Learners

A multimodal learning research project exploring learner engagement and personalization using eye-related signals and adaptive learning approaches.

## Project Structure

- `Pupil-dilation/` — pupil-dilation analysis and API.
- `personalisation/` — personalization experiments and code.
- `eye-stream-on-daisee/` — eye-stream research experiments.
- `Fixation-Duration_Saccade-Velocity/` — fixation and saccade analysis.
- `cmose-few-shot-personalisation/` — few-shot personalization experiments.
- `models/` — model-related files.
- `frontend/` — React, TypeScript, and Vite user interface.

## Requirements

- Python 3
- Node.js and npm
- A webcam for live webcam features

## Run the Backend

Open Terminal and navigate to the project root:

```bash
cd Pupil-dilation
python3 -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn opencv-python numpy
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Keep this Terminal open while using the UI.

## Run the Frontend

Open a second Terminal window and navigate to the project root:

```bash
cd frontend
npm install
npm run dev -- --host 127.0.0.1
```

Open the local URL printed by Vite in your browser.

## Webcam API

The frontend webcam panel communicates with the backend endpoint:

`POST http://127.0.0.1:8000/analyze-frame`

Allow webcam access in your browser when prompted.

## Notes

- The webcam feature requires the backend to be running.
- Keep `.env` files, virtual environments, dependency folders, and private data out of Git.
- Research folders may have their own dependencies and setup instructions.
- This repository contains research experiments; individual components may require additional datasets or configuration.


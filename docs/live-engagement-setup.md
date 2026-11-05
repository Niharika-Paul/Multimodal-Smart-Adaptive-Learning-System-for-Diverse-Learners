# Live dashboard setup

The dashboard and webcam analysis API are both in this repository. The frontend uses `VITE_API_BASE_URL` to choose the backend origin; the default is `http://127.0.0.1:8001`. The API allows local Vite origins on ports 5173, 8080, 8081, and 8082. For another frontend origin, set `FRONTEND_ORIGINS` to a comma-separated list of full origins.

## External model assets

The following assets are intentionally excluded from Git:

- The upstream [`video_features`](https://github.com/v-iashin/video_features) library and its pretrained RGB I3D weights.
- The exported CMOSE W0-P checkpoint archive. `W0P_CHECKPOINT_DIR` must point to its extracted root directory, containing `w0p_checkpoints/` and `w0_base_checkpoints/`. The integration loads `w0p_checkpoints/fold_1__video14_person0__random__K20.pt` and its referenced W0 base checkpoint.
- `I3D_Feature_Extraction.ipynb` and `final_data_1.json` are training workflow/data artifacts and are not needed to run the live service.

Obtain the upstream library and trained checkpoint assets through the project’s authorized research storage. Do not commit checkpoint files, training data, student recordings, or webcam captures. The I3D library's own environment instructions (`conda_env.yml`) may be used to install its additional dependencies; the backend's direct service dependencies are listed in `Pupil-dilation/requirements.txt`.

## Install

Use Python 3.10 or newer and Node.js/npm. Install PyTorch and TorchVision builds compatible with the target CPU/GPU platform using the [official PyTorch install selector](https://pytorch.org/get-started/locally/), then install the service requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r Pupil-dilation/requirements.txt
```

Copy `frontend/.env.example` to `frontend/.env.local` and edit the backend URL if needed. Set model paths in the shell that will start the backend:

```bash
export I3D_LIBRARY_DIR="/absolute/path/to/video_features"
export W0P_CHECKPOINT_DIR="/absolute/path/to/CMOSE_WOP_checkpoints_export"
export FRONTEND_ORIGINS="http://localhost:8080,http://127.0.0.1:8080"
```

`I3D_LIBRARY_DIR` must contain `models/i3d/checkpoints/i3d_rgb.pt` and `configs/i3d.yml`. `W0P_CHECKPOINT_DIR` must contain the checkpoint folders described above. The MediaPipe face landmarker model is already included at `Pupil-dilation/face_landmarker.task`.

## Run

Terminal 1:

```bash
cd Pupil-dilation
python -m uvicorn api:app --host 127.0.0.1 --port 8001
```

Terminal 2:

```bash
cd frontend
npm ci
npm run dev -- --host 127.0.0.1 --port 8080
```

Open `http://localhost:8080`, allow camera access, and keep both processes running. The API health route is `http://127.0.0.1:8001/health`. A live I3D feature is produced after 65 frames sampled at 2 fps; inference then uses the trained W0-P classifier. The model needs about 32.5 seconds to collect each clip, in addition to extraction time.

## Validate

From the repository root, run:

```bash
(cd frontend && npm ci && npm run build)
python3 -m py_compile Pupil-dilation/api.py Pupil-dilation/live_i3d.py Pupil-dilation/model_inference.py Pupil-dilation/eye_cropper.py
curl -i http://127.0.0.1:8001/health
curl -i -X OPTIONS http://127.0.0.1:8001/analyze-frame \
  -H 'Origin: http://localhost:8080' \
  -H 'Access-Control-Request-Method: POST' \
  -H 'Access-Control-Request-Headers: content-type'
```

The webcam endpoint accepts JSON `{ "image": "data:image/jpeg;base64,...", "session_id": "..." }` at `POST /analyze-frame`. It returns face status, blink/gaze metrics, and the current engagement collection/inference state. With assets configured, allow the webcam to run until the first clip is processed to verify an actual W0-P prediction.

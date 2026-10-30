# Learning Dashboard Frontend

React/Vite single-page application for the Multimodal Smart Adaptive Learning System. It provides the student-facing dashboard and learning, analytics, and settings pages. The dashboard’s webcam component sends frames to the FastAPI service; live eye metrics and W0-P predictions are rendered from its responses.

For the whole application—including backend installation, external I3D/W0-P assets, API details, privacy notes, and research limitations—see the [project README](../README.md) and [live engagement setup guide](../docs/live-engagement-setup.md).

## Contents

- [Pages and features](#pages-and-features)
- [Technology](#technology)
- [Install and run](#install-and-run)
- [Environment configuration](#environment-configuration)
- [Backend integration](#backend-integration)
- [Source structure](#source-structure)
- [Tests and lint](#tests-and-lint)
- [Troubleshooting](#troubleshooting)
- [Implementation boundaries](#implementation-boundaries)

## Pages and features

Routes are defined in `src/App.tsx` and use React Router:

| Route | Page | Current behavior |
| --- | --- | --- |
| `/` | Dashboard | Live webcam/API state and eye/engagement metric cards, plus sample charts, summary, and feedback panels. |
| `/learning` | Learning | Course progress cards and recommended items; the values are presentational sample data. |
| `/analytics` | Analytics | Engagement/focus charts and study summary; chart data is static sample data. |
| `/settings` | Settings | Profile and preference controls; most toggle state is local to the mounted page and is not persisted by a backend. |

The shared top navigation links to each page. Unknown routes render the not-found page.

### Live dashboard components

- **`WebcamPanel`** requests browser camera access, displays the local feed, and reports camera/API status. It captures frames at approximately 640×480, JPEG-encodes them, and sends requests to the backend while the camera is active.
- **`EngagementMetrics`** displays model confidence and class, clip collection status, estimated screen gaze/eye-contact percentage, blink count/rate, and a normalized iris ratio when the backend returns those measurements.
- **`EngagementChart`, `SessionSummary`, `AdaptiveFeedback`, and `ExplainableAI`** currently show static example values/text; they are not populated from a stored session history or adaptive-learning service.
- **`MultimodalModules`** marks eye tracking and engagement detection as available in the UI. Audio analysis and handwriting analysis are marked “Coming soon” and do not have active processing pipelines.

## Technology

The versions below are the ranges declared in `package.json` (the lockfile resolves exact installs):

- React and React DOM `^18.3.1`
- TypeScript `^5.8.3`
- Vite `^8.0.0` with `@vitejs/plugin-react` `^6.0.0`
- Tailwind CSS `^3.4.17`
- React Router DOM `^6.30.1`
- Radix UI primitives and shadcn-style components
- Framer Motion `^12.38.0` for UI transitions
- Recharts `^2.15.4` for charts
- Lucide React `^0.462.0` for icons
- TanStack React Query `^5.83.0`, React Hook Form, Zod, and supporting UI utilities
- Vitest `^4.1.0`, Testing Library, and ESLint for checks

Use a Node.js/npm version supported by Vite 8. The frontend is private and is not published as a standalone npm package.

## Install and run

From the repository root:

```bash
cd frontend
cp .env.example .env.local
npm ci
npm run dev
```

The Vite configuration uses port `8080`. Open [http://localhost:8080](http://localhost:8080). If port 8080 is occupied, start Vite on a different port and add that exact browser origin to the backend’s `FRONTEND_ORIGINS` setting.

Build and serve a production preview:

```bash
npm run build
npm run preview
```

Run the frontend test suite and linter:

```bash
npm test
npm run lint
```

`npm test` runs the Vitest suite. The repository currently has one frontend test. `npm run lint` is available but currently reports existing lint errors and warnings; see [Tests and lint](#tests-and-lint).

## Environment configuration

`VITE_API_BASE_URL` is the backend origin without an API route. The example file sets:

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8001
```

Change it in `.env.local` for a different backend, for example:

```dotenv
VITE_API_BASE_URL=http://localhost:8001
```

The webcam component appends `/analyze-frame` and strips a trailing slash. Restart the Vite dev server after changing `.env.local`; for production builds, the value is incorporated at build time. The backend must allow the page’s exact origin through `FRONTEND_ORIGINS`; `localhost` and `127.0.0.1` are distinct origins.

## Backend integration

The live panel uses `navigator.mediaDevices.getUserMedia` and requests video without audio. The captured frame is encoded as a JPEG data URI and sent as JSON to `POST ${VITE_API_BASE_URL}/analyze-frame`:

```json
{
  "image": "data:image/jpeg;base64,...",
  "session_id": "unique-browser-session"
}
```

The `session_id` lets the backend reset its I3D frame buffer when a browser session changes. Requests are serialized in the component so a slow response does not create overlapping frame requests; each request has a 15-second timeout.

The response includes face status, blink count and (after a complete observation window) blink rate, gaze/estimated eye-contact values, normalized iris ratio, and engagement collection/inference status. Once 65 RGB samples have been collected at 2 fps and I3D extraction completes, it can include the W0-P label, class index, confidence, class probabilities, model name, and checkpoint metadata. `EngagementMetrics` maps those values to the metric cards. The backend’s `/health` route is useful for a service check but does not test webcam access or model inference.

When the API request fails or returns a non-2xx response, the webcam panel displays **API response error** and the error message. The frame loop continues to attempt later requests while the camera is active. A prediction is not fabricated when the service is unavailable.

The backend’s external asset setup and endpoints are documented in the [root project README](../README.md#external-model-assets) and [setup guide](../docs/live-engagement-setup.md#external-model-assets).

## Source structure

```text
frontend/
├── public/                  # Static assets
├── src/
│   ├── App.tsx              # Router and app-level providers
│   ├── main.tsx             # React entry point
│   ├── pages/               # Dashboard, learning, analytics, settings, 404
│   ├── components/
│   │   ├── dashboard/       # Webcam, metrics, charts, feedback, navigation
│   │   ├── ui/              # Reusable Radix/shadcn-style controls
│   │   └── NavLink.tsx      # Navigation link component
│   ├── hooks/               # Toast and responsive-layout hooks
│   ├── lib/                 # Shared utilities
│   ├── test/                # Vitest setup and example test
│   ├── index.css            # Tailwind layers and global styles
│   └── vite-env.d.ts        # Vite environment typings
├── .env.example             # Backend URL example
├── package.json             # Scripts and dependency ranges
├── package-lock.json        # Reproducible npm dependency lock
└── vite.config.ts           # React plugin, alias, dev server port
```

Key dashboard files include `src/pages/Index.tsx`, `src/components/dashboard/WebcamPanel.tsx`, and `src/components/dashboard/EngagementMetrics.tsx`.

## Tests and lint

The test command is `npm test` (`vitest run`). Run it from `frontend/` after `npm ci`.

The current `npm run lint` command reports 6 errors and 7 warnings in existing source/config files. Errors include explicit `any` types in chart tooltips, empty interfaces in UI components, and a CommonJS `require` in the Tailwind config. The dashboard build and targeted lint checks are separate from the full lint command; this README does not claim full lint passes.

## Troubleshooting

- **`Failed to fetch` / API response error:** confirm the backend is running at `VITE_API_BASE_URL`, check `http://127.0.0.1:8001/health`, and inspect the browser Network panel for the request to `/analyze-frame`.
- **CORS or `OPTIONS` returns 400:** configure `FRONTEND_ORIGINS` in the backend shell with the page origin including scheme, host, and port, then restart the backend.
- **Webcam permission denied or unavailable:** allow camera access for the page in browser settings, use localhost/HTTPS, and close other applications that may be holding the camera.
- **Page loads but shows “Waiting for face”:** check lighting, camera framing, backend logs, and that `Pupil-dilation/face_landmarker.task` exists in the backend working directory.
- **Still collecting frames:** the I3D path waits for 65 samples at 2 fps. Extraction follows collection; a prediction is not immediate when the dashboard opens.
- **Missing I3D/W0-P files:** configure `I3D_LIBRARY_DIR` and `W0P_CHECKPOINT_DIR` for the backend process. The model library and trained checkpoints are not included in this frontend or repository checkout by default.
- **Environment change ignored:** restart Vite after editing `.env.local`; Vite variables are injected when the dev server/build starts.
- **`npm ci` fails:** use the committed `package-lock.json` and a Node/npm version supported by Vite 8. Avoid changing the lockfile to work around local installation issues without reviewing the dependency change.

## Implementation boundaries

Live eye and engagement metrics are API-backed. The learning page, analytics history, session summary, adaptive feedback copy, and several settings are sample/local UI and are not stored or synchronized with a learner service. Estimated eye contact is a heuristic based on iris position; it is not a calibrated gaze measurement. See the [root README limitations and privacy sections](../README.md#limitations) for model/generalization and webcam-processing context.

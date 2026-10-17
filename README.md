# Stillpoint AI

**A local-first interview-session dashboard and Google Meet companion prototype.**

Stillpoint AI brings webcam or shared-screen video into an interviewer dashboard with experimental visual telemetry, a live event timeline, and downloadable session reports. An optional Chrome extension adds a compact companion HUD to Google Meet.

> **Prototype notice:** Stillpoint is an experimental software project, not a validated assessment, lie detector, or measure of a person's attention, honesty, mood, or suitability for a role. Its signals and scores can be inaccurate or biased. Do not use them to make or justify hiring, disciplinary, academic, or other consequential decisions. Obtain informed consent before using a camera or collecting session data.

## Features

- **Three video sources:** webcam, a browser-selected screen/window/tab (for example, a Meet call), or the extension bridge.
- **In-browser visual processing:** MediaPipe Face Mesh and TensorFlow.js COCO-SSD are used for experimental face, gaze, posture, and object telemetry.
- **Experimental motion signals:** gaze-pattern heuristics and sampled-frame comparisons estimate repeated eye movement, gestures, and background motion; these do not establish reading, cheating, or intent.
- **Live session dashboard:** video overlay, telemetry cards, and a chronological event timeline.
- **Interviewer controls:** calibrate, flag a moment, take a snapshot, and add a note.
- **Session reports:** a multi-signal summary, JSON audit export, and print/save as PDF.
- **Optional Google Meet companion:** a draggable HUD injected into Meet, with a link to the local dashboard.
- **No build step or npm packages:** the local server uses Node.js built-in modules.

## Four proposed pioneer factors

The system documentary describes the following ideas as “world-first” factors. That novelty claim has not been independently verified; the points below describe the current prototype, not validated or guaranteed capabilities:

1. **In-browser dual-model processing:** MediaPipe Face Mesh and TensorFlow.js COCO-SSD run in the browser for face-landmark and object telemetry. This prototype loads libraries and model assets from jsDelivr, so model processing is local but initial loading still needs an internet connection.
2. **Eye-movement pattern heuristic:** the prototype tracks gaze estimates and flags some repeated horizontal eye movements. It cannot determine whether someone is reading AI-generated text, using a teleprompter, or cheating.
3. **Spatial motion estimates:** sampled video frames are compared across face, gesture, and background regions to estimate motion. These heuristics can misclassify ordinary movement or changes in lighting; they do not prove an intrusion or policy violation.
4. **No automatic video-upload service:** routine frame analysis is performed in the browser, and the app does not provide a remote video-processing endpoint. Session metrics and event data are sent to the local Node.js server for report generation. If an interviewer takes a snapshot, it remains in the active page session and can appear in the on-screen report.

## Requirements

- Node.js (18 or later recommended)
- A current browser with camera and screen-capture support (Chrome or Edge recommended)
- Internet access on first load so the browser can download the MediaPipe, TensorFlow.js, and model assets referenced by the page
- A camera for webcam sessions; screen/window/tab capture requires a browser permission prompt

## Run locally

From the project directory:

```powershell
node server.js
```

Open [http://localhost:8000](http://localhost:8000). To use a different port:

```powershell
$env:PORT = 8080
node server.js
```

Then open `http://localhost:8080`. Stop the server with `Ctrl+C`.

Camera and screen-capture permissions are granted by the browser when a session starts. When using screen capture, select the intended tab, window, or display in the browser picker.

## Chrome / Edge extension

1. Start the local dashboard using the instructions above.
2. Open `chrome://extensions` in Chrome or `edge://extensions` in Edge.
3. Enable **Developer mode**.
4. Choose **Load unpacked** and select this project's `extension` folder.
5. Open or reload a page on [meet.google.com](https://meet.google.com). The extension injects its companion HUD into Meet.
6. Use **Full Suite** in the HUD or the extension popup to open the dashboard at `http://localhost:8000`.

The extension is an optional companion; the dashboard can also be used without installing it.

## Technical Documentation & Whitepaper

For an in-depth breakdown of the mathematical models, computer vision pipelines, SolvePnP Euler angle computations, saccadic reading algorithms, and privacy architecture, see:
- 📄 **[Stillpoint AI System Documentary (PDF)](./Stillpoint_AI_System_Documentary.pdf)** — Publication-ready technical whitepaper & system blueprint.
- 🌐 **[Stillpoint AI System Documentary (HTML)](./Stillpoint_AI_System_Documentary.html)** — Interactive, printable A4 documentation source.

## Project layout

| Path | Purpose |
| --- | --- |
| `Stillpoint_AI_System_Documentary.pdf` | Comprehensive system documentary, mathematical models, and whitepaper |
| `Stillpoint_AI_System_Documentary.html` | Printable HTML publication source for the system documentary |
| `index.html`, `styles.css` | Dashboard structure and presentation |
| `app.js` | Session controls, source selection, timeline, snapshots, and reports |
| `ml-engine.js` | Browser-side MediaPipe and TensorFlow.js integration |
| `ai-evaluator.js` | Heuristic evaluation summary generation |
| `server.js` | Local static-file server and evaluation API |
| `extension/` | Manifest V3 Meet companion extension |

## Privacy and responsible use

- Video processing and model inference are designed to run in the browser. The application does not provide a remote video-upload or video-processing service.
- The dashboard sends session summary metrics and event data to its local Node server at `POST /api/analyze-evaluation` to generate a report. Manually captured snapshots are retained in the active page session and can be displayed in the report. Review the payload and source code before using the app with sensitive information.
- The dashboard loads third-party scripts and model files from jsDelivr. Those requests require internet access and disclose ordinary connection metadata to the hosting provider.
- The extension requests access to `meet.google.com` and injects its HUD into matching Meet pages. Review its permissions and behavior before installing it.
- Use only with participants' informed consent, follow applicable privacy and employment laws, and provide a non-camera alternative where appropriate.
- Visual signals are context-dependent and can fail under different lighting, camera positions, disabilities, neurodivergence, skin tones, and other conditions. Treat all output as unverified prompts for human review, not objective facts.

## Troubleshooting

- **Camera or screen capture is unavailable:** use a current browser and run the dashboard from `localhost` (not by opening `index.html` as a file). Check the browser's site permissions and operating-system camera permissions.
- **AI models do not initialize:** confirm internet access and check whether a network filter is blocking jsDelivr. The model libraries and assets are loaded from the CDN.
- **The extension HUD is missing:** confirm Developer mode is enabled, the `extension` folder is loaded, and the Meet tab was reloaded after installation.
- **The extension cannot open the dashboard:** make sure `node server.js` is running on the same computer and port expected by the extension (`8000`).

## Development notes

This project currently has no package manifest, automated test suite, or build pipeline. The local server and evaluation endpoint use Node.js built-ins; browser-side ML libraries are included through CDN script tags in `index.html`.

## License

No license is currently provided. Unless a license is added, all rights remain with the copyright holder; obtain permission before redistributing or reusing this project.

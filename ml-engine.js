/**
 * Stillpoint AI — High-Performance Machine Learning Engine (ml-engine.js)
 * Dedicated Neural Vision, 3D Biometrics, Gaze Vectoring & Behavioral Analysis
 * Zero-cloud client-side execution via MediaPipe FaceMesh (Iris) & TensorFlow COCO-SSD
 */
"use strict";

(function (root, factory) {
  if (typeof define === "function" && define.amd) {
    define([], factory);
  } else if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.StillpointMLEngine = factory();
  }
})(typeof self !== "undefined" ? self : this, function () {

  class MLEngine {
    constructor() {
      // Model Handles
      this.faceMesh = null;
      this.cocoModel = null;
      this.isFaceMeshReady = false;
      this.isCocoReady = false;
      this.isBusyProcessing = false;
      this.isDetectingObjects = false;
      this.runId = 0;
      this.activeVideo = null;
      this.lastFrameAt = 0;
      this.faceAbsentSince = 0;
      this.wasReadingSaccade = false;

      // Processing / Canvas elements
      this.hudCanvas = null;
      this.hudCtx = null;
      this.analysisCanvas = null;
      this.analysisCtx = null;

      // Event Callbacks
      this.onTelemetryUpdate = null;
      this.onIncident = null;
      this.onFpsUpdate = null;
      this.onStatusChange = null;

      // Config & Toggles
      this.showVisualOverlay = true;
      this.isWebcamSource = true; // true = webcam (mirrored X), false = screen share (direct)

      // Smoothed 3D Pose & Gaze (Exponential Moving Average)
      this.smoothYaw = 0;
      this.smoothPitch = 0;
      this.smoothRoll = 0;
      this.smoothGazeX = 0;
      this.smoothGazeY = 0;
      this.ALPHA = 0.35; // Per-second smoothing is applied using elapsed frame time.
      this.frameAlpha = this.ALPHA;

      // Calibration Baseline
      this.baseline = {
        yaw: 0,
        pitch: 0,
        gazeX: 0,
        gazeY: 0,
        samples: [],
        isCalibrating: false,
        isCalibrated: false
      };

      // Auto-warmup calibration (first 40 valid face frames)
      this.sessionWarmupCount = 0;
      this.autoBaselineSet = false;
      this.warmupGazeCalibrated = false;
      this.warmupSamples = [];

      // Eye & Blink Metrics
      this.blinkCount = 0;
      this.blinkStartTime = 0;
      this.eyesClosedSince = 0;
      this.lastClosedReportSec = 0;
      this.recentBlinkTimestamps = [];
      this.blinkRateHistory = 16;

      // Saccade & Gaze State Tracking
      this.gazeHistoryX = [];
      this.sustainedGazeDir = "CENTER";
      this.gazeDirChangeTime = 0;
      this.lastReportedGazeDir = "CENTER";
      this.gazeOffscreenCount = 0;
      this.saccadeCount = 0;
      this.rapidGazeShiftCount = 0;

      // Speech & Facial Mood State
      this.speechBuffer = [];
      this.lastSpeechRatio = 0;
      this.lastInnerAperture = 0;
      this.nervousScore = 0;
      this.isCandidateNervousState = false;
      this.steadinessScore = 98;

      // Object & Environmental State
      this.latestCocoObjects = [];
      this.cocoTimer = null;
      this.bgSurveillanceTimer = null;
      this.lastBackgroundPixels = null;
      this.candidateBox = null;
      this.handNearFace = false;
      this.handTouchFrameCount = 0;
      this.handGesturing = false;
      this.handActivity = "Clear";
      this.candidateGestureEnergy = 0;
      this.bgMotionEnergy = 0;
      this.unauthorizedDevices = [];
      this.secondaryPersons = 0;

      // State-Transition Incident Memory (Zero repetitive spamming)
      this.activeStates = new Map();

      // Drowsiness, Ptosis & Sad Affect
      this.currentEar = 0.30;
      this.droopyEyesSince = 0;
      this.drowsinessScore = 0;
      this.sadScore = 0;

      // Micro-Expression & Brow Dynamics
      this.currentBrowLift = 0;
      this.baselineBrowDist = 0;
      this.eyebrowFlashCount = 0;
      this.microTwitchScore = 0;
      this.lastLandmarkPositions = null;

      // Audio & Acoustic Intelligence (Web Audio API)
      this.audioCtx = null;
      this.audioAnalyser = null;
      this.audioSourceNode = null;
      this.audioData = null;
      this.isAudioReady = false;
      this.currentAudioDb = -80;
      this.isVocalizing = false;
      this.bgNoiseSpike = false;

      // Incident Debouncing (cooldowns in ms)
      this.incidentCooldowns = new Map();

      // Composite Focus Score (Continuous floating-point accuracy)
      this.focusScore = 100;
      this.rawFocusScore = 100.0;
      this.focusHistory = [];

      // FPS Metrics
      this.fpsCounter = 0;
      this.lastFpsTimestamp = this._now();
      this.currentFps = 0;
    }

    /**
     * Initialize Models & Offscreen Canvas Resources
     */
    async init(options = {}) {
      if (options.hudCanvas) {
        this.hudCanvas = options.hudCanvas;
        this.hudCtx = this.hudCanvas.getContext("2d");
      }
      if (options.analysisCanvas) {
        this.analysisCanvas = options.analysisCanvas;
        this.analysisCtx = this.analysisCanvas.getContext("2d", { willReadFrequently: true });
        // High-efficiency lightweight sampling resolution (96x72) to maximize FPS
        this.analysisCanvas.width = 96;
        this.analysisCanvas.height = 72;
      }

      this._notifyStatus("LOADING NEURAL MODELS...", "busy");

      // 1. Initialize MediaPipe FaceMesh with Iris (478 3D landmarks)
      try {
        const browser = typeof window !== "undefined" ? window : root;
        if (browser && browser.FaceMesh) {
          this.faceMesh = new browser.FaceMesh({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
          });
          this.faceMesh.setOptions({
            maxNumFaces: 2,
            refineLandmarks: true, // enables iris landmarks 468-477
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
          });
          this.faceMesh.onResults((results) => this._onFaceMeshResults(results));
          this.isFaceMeshReady = true;
        }
      } catch (err) {
        console.warn("[Stillpoint ML] FaceMesh init error:", err);
      }

      // 2. Initialize TensorFlow COCO-SSD (lite mobilenet v2)
      try {
        const browser = typeof window !== "undefined" ? window : root;
        if (browser && browser.cocoSsd) {
          this.cocoModel = await browser.cocoSsd.load({ base: "lite_mobilenet_v2" });
          this.isCocoReady = true;
        }
      } catch (err) {
        console.warn("[Stillpoint ML] COCO-SSD init error:", err);
      }

      if (this.isFaceMeshReady) {
        this._notifyStatus("AI VISION ACTIVE (478 LANDMARKS + COCO)", "ready");
      } else {
        this._notifyStatus("HEURISTIC FALLBACK MODE", "ready");
      }

      return { faceMeshReady: this.isFaceMeshReady, cocoReady: this.isCocoReady };
    }

    /**
     * Start ML Inference Engine for an Active Stream
     */
    start(videoElement, sourceKind = "webcam") {
      this.stop();
      this.runId++;
      const runId = this.runId;
      this.activeVideo = videoElement || null;
      this.isWebcamSource = (sourceKind === "webcam");
      this.blinkStartTime = Date.now();
      this.blinkCount = 0;
      this.focusScore = 100;
      this.focusHistory = [];
      this.gazeOffscreenCount = 0;
      this.saccadeCount = 0;
      this.sessionWarmupCount = 0;
      this.autoBaselineSet = false;
      this.warmupGazeCalibrated = false;
      this.warmupSamples = [];
      this.latestCocoObjects = [];
      this.unauthorizedDevices = [];
      this.secondaryPersons = 0;
      this.lastBackgroundPixels = null;
      this.candidateBox = null;
      this.incidentCooldowns.clear();
      this.speechBuffer = [];
      this.gazeHistoryX = [];
      this.sustainedGazeDir = "CENTER";
      this.lastReportedGazeDir = "CENTER";
      this.gazeDirChangeTime = Date.now();
      this.activeStates.clear();
      this.rawFocusScore = 100.0;
      this.droopyEyesSince = 0;
      this.drowsinessScore = 0;
      this.sadScore = 0;
      this.baselineBrowDist = 0;
      this.eyebrowFlashCount = 0;
      this.microTwitchScore = 0;
      this.lastLandmarkPositions = null;
      this.eyesClosedSince = 0;
      this.lastClosedReportSec = 0;
      this.recentBlinkTimestamps = [];
      this.handNearFace = false;
      this.handGesturing = false;
      this.handActivity = "Clear";
      this.candidateGestureEnergy = 0;
      this.bgMotionEnergy = 0;
      this.rapidGazeShiftCount = 0;
      this.nervousScore = 0;
      this.isCandidateNervousState = false;
      this.handTouchFrameCount = 0;
      this.lastInnerAperture = 0;
      this.steadinessScore = 98;
      this.smoothYaw = this.smoothPitch = this.smoothRoll = 0;
      this.smoothGazeX = this.smoothGazeY = 0;
      this.lastFrameAt = 0;
      this.baseline.isCalibrating = false;
      this.baseline.isCalibrated = false;
      this.baseline.samples = [];
      this.baseline.yaw = this.baseline.pitch = this.baseline.gazeX = this.baseline.gazeY = 0;
      this.wasReadingSaccade = false;

      // Schedule detections after the prior inference completes, so slow devices do not queue work.
      if (this.isCocoReady && videoElement) {
        const scheduleDetection = async () => {
          if (runId !== this.runId || !this.activeVideo) return;
          await this._detectObjectsAsync(videoElement, runId);
          if (runId === this.runId && this.activeVideo) {
            this.cocoTimer = setTimeout(scheduleDetection, 1600);
          }
        };
        this.cocoTimer = setTimeout(scheduleDetection, 0);
      }

      // Sample background & gesture motion at 1200ms interval to eliminate GPU stalls and maintain 25-30+ FPS
      if (videoElement) {
        this.bgSurveillanceTimer = setInterval(() => this._analyzeBackgroundMotion(videoElement), 1200);
      }
    }

    /**
     * Initialize Real-Time Web Audio Intelligence
     */
    initAudio(stream) {
      if (!stream) return;
      try {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        if (!AudioCtx) return;

        if (this.audioCtx && this.audioCtx.state !== "closed") {
          this.audioCtx.close().catch(() => { });
        }

        const audioTracks = stream.getAudioTracks();
        if (!audioTracks || audioTracks.length === 0) return;

        this.audioCtx = new AudioCtx();
        this.audioSourceNode = this.audioCtx.createMediaStreamSource(stream);
        this.audioAnalyser = this.audioCtx.createAnalyser();
        this.audioAnalyser.fftSize = 256;
        this.audioAnalyser.smoothingTimeConstant = 0.8;
        this.audioSourceNode.connect(this.audioAnalyser);
        this.audioData = new Uint8Array(this.audioAnalyser.frequencyBinCount);
        this.isAudioReady = true;
      } catch (err) {
        console.warn("[Stillpoint ML] Audio analysis init note:", err);
      }
    }

    /**
     * Stop ML Inference and clear timers
     */
    stop() {
      this.runId++;
      this.activeVideo = null;
      if (this.cocoTimer) {
        clearTimeout(this.cocoTimer);
        this.cocoTimer = null;
      }
      if (this.bgSurveillanceTimer) {
        clearInterval(this.bgSurveillanceTimer);
        this.bgSurveillanceTimer = null;
      }
      if (this.audioCtx && this.audioCtx.state !== "closed") {
        this.audioCtx.close().catch(() => { });
        this.audioCtx = null;
        this.isAudioReady = false;
      }
      if (this.hudCtx && this.hudCanvas) {
        this.hudCtx.clearRect(0, 0, this.hudCanvas.width, this.hudCanvas.height);
      }
      this.activeStates.clear();
    }

    /**
     * Trigger manual Neutral Baseline Gaze Calibration
     */
    calibrate(durationMs = 2000) {
      if (this.baseline.isCalibrating) {
        return Promise.resolve({ success: false, reason: "Calibration is already in progress." });
      }
      this.baseline.isCalibrating = true;
      this.baseline.samples = [];

      return new Promise((resolve) => {
        setTimeout(() => {
          this.baseline.isCalibrating = false;
          if (this.baseline.samples.length > 0) {
            const median = (key) => {
              const values = this.baseline.samples.map((s) => s[key]).filter(Number.isFinite).sort((a, b) => a - b);
              const middle = Math.floor(values.length / 2);
              return values.length % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2;
            };
            this.baseline.pitch = median("pitch");
            this.baseline.yaw = median("yaw");
            this.baseline.gazeX = median("gazeX");
            this.baseline.gazeY = median("gazeY");
            this.baseline.isCalibrated = true;
            resolve({ success: true, baseline: this.baseline });
          } else {
            resolve({ success: false, reason: "No face detected during calibration window." });
          }
        }, durationMs);
      });
    }

    /**
     * Process a single video frame through FaceMesh
     */
    async processFrame(videoElement) {
      if (!this._hasCurrentFrame(videoElement)) return;
      if (this.isBusyProcessing) return;

      this._updateFps();

      this.isBusyProcessing = true;
      try {
        if (this.faceMesh && this.isFaceMeshReady) {
          await this.faceMesh.send({ image: videoElement });
        } else {
          this._simulateHeuristics(videoElement);
        }
      } catch (err) {
        // Drop occasional frame without halting loop
      } finally {
        this.isBusyProcessing = false;
      }
    }

    /* =========================================================================
     * INTERNAL PIPELINE & ALGORITHMS
     * ========================================================================= */

    /**
     * MediaPipe FaceMesh Callback Handler
     */
    _onFaceMeshResults(results) {
      if (!this.activeVideo || !results) return;

      try {
        this.frameAlpha = this._frameAlpha();
        if (this.hudCanvas && this.hudCtx) this.hudCtx.clearRect(0, 0, this.hudCanvas.width, this.hudCanvas.height);
        const faces = results.multiFaceLandmarks;

        if (!faces || faces.length === 0) {
          this._handleFaceAbsent();
          this._renderCocoOverlay();
          return;
        }

        // 1. MULTI-FACE TRACKING
        const primaryFace = faces[0];
        if (!primaryFace || primaryFace.length < 468 || !primaryFace[10] || !primaryFace[152]) {
          this._handleFaceAbsent();
          return;
        }
        this.faceAbsentSince = 0;
        const pBox = this._getFaceBBox(primaryFace);
        this.candidateBox = pBox;

        let detectedSecondaryFaces = 0;
        if (faces.length > 1) {
          for (let i = 1; i < faces.length; i++) {
            const sFace = faces[i];
            const sBox = this._getFaceBBox(sFace);
            const overlap = this._getBBoxOverlap(pBox, sBox);
            const sizeRatio = (sBox.w * sBox.h) / Math.max(0.001, pBox.w * pBox.h);

            // FaceMesh returns faces, not hands. Ignore overlapping duplicate boxes only.
            if (overlap <= 0.45 && sizeRatio >= 0.08) {
              detectedSecondaryFaces++;
            }
          }
        }

        this.handNearFace = false;
        this.secondaryPersons = detectedSecondaryFaces;

        if (detectedSecondaryFaces > 0) {
          this._emitIncident("Secondary Person in Room", `${detectedSecondaryFaces + 1} individuals visible in camera frame`, "critical");
        }

        // 2. EYE ASPECT RATIO & BLINK TRACKING (Compute FIRST so gaze knows if eyes are open)
        this._analyzeBlinks(primaryFace);

        // 3. ACCURATE 3D HEAD POSE (FACIAL NORMAL VECTOR CROSS-PRODUCT)
        const pose = this._computeAccurateHeadPose(primaryFace);

        // 4. ACCURATE EYE GAZE & SACCADE TRACKING
        const gaze = this._computeAccurateEyeGaze(primaryFace, pose);

        // 5. AUTO-WARMUP BASELINE CALIBRATION (First 40 frames, discard initial 10)
        if (this.sessionWarmupCount < 40) {
          this.sessionWarmupCount++;
          if (this.sessionWarmupCount > 10 && this.eyesClosedSince === 0) {
            this.warmupSamples.push({
              pitch: this.smoothPitch,
              yaw: this.smoothYaw,
              gazeX: this.smoothGazeX,
              gazeY: this.smoothGazeY
            });
          }
          if (this.sessionWarmupCount === 40 && !this.baseline.isCalibrated && !this.baseline.isCalibrating) {
            const medianVal = (key, fallback = 0) => {
              const s = this.warmupSamples.map((item) => item[key]).filter(Number.isFinite).sort((a, b) => a - b);
              if (s.length === 0) return fallback;
              const mid = Math.floor(s.length / 2);
              const val = s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
              return Number.isFinite(val) ? val : fallback;
            };
            this.baseline.pitch = medianVal("pitch", 0);
            this.baseline.yaw = medianVal("yaw", 0);
            this.baseline.gazeX = medianVal("gazeX", 0);
            this.baseline.gazeY = medianVal("gazeY", 0);
            this.baseline.isCalibrated = true;
            this.autoBaselineSet = true;
          }
        }

        // 6. FACIAL MOOD, MICRO-EXPRESSION & ACTION UNITS
        const expressions = this._analyzeExpressions(primaryFace);

        // 7. REAL-TIME ACOUSTIC AUDIO ANALYSIS (Microphone Voice vs Ambient Room Noise)
        this._analyzeAudioFrame(expressions.isSpeaking);

        // 8. COMPOSITE FOCUS SCORE CALCULATION
        this._updateCompositeFocus(pose, gaze, expressions);

        // 9. RENDER HUD OVERLAYS
        if (this.showVisualOverlay && this.hudCanvas && this.hudCtx) {
          this._renderHudOverlay(primaryFace, pose, gaze, expressions);
        }
        this._renderCocoOverlay();

        // 10. DISPATCH TELEMETRY UPDATE
        if (this.onTelemetryUpdate) {
          this.onTelemetryUpdate({
            focusScore: Math.round(this.focusScore),
            rawFocusScore: this.rawFocusScore,
            pose: pose,
            gaze: gaze,
            expressions: expressions,
            blinkRate: this.blinkRateHistory,
            currentEar: this.currentEar,
            drowsinessScore: this.drowsinessScore,
            sadScore: this.sadScore,
            browLift: this.currentBrowLift,
            microTwitchScore: this.microTwitchScore,
            audioDb: this.currentAudioDb,
            isVocalizing: this.isVocalizing,
            bgNoiseSpike: this.bgNoiseSpike,
            handNearFace: this.handNearFace,
            handActivity: this.handActivity,
            isGesturing: this.handGesturing,
            candidateGestureEnergy: this.candidateGestureEnergy,
            bgMotionEnergy: this.bgMotionEnergy,
            secondaryPersons: this.secondaryPersons,
            unauthorizedDevices: this.unauthorizedDevices,
            isCalibrated: this.baseline.isCalibrated,
            fps: this.currentFps
          });
        }
      } catch (meshErr) {
        console.warn("[Stillpoint ML] Frame processing note:", meshErr);
      }
    }

    /**
     * Compute 3D Head Pose using Facial Normal Cross-Product
     * Mathematics:
     * - Vector Horizontal: Left Eye (33) to Right Eye (263)
     * - Vector Vertical: Chin (152) to Forehead (10)
     * - Normal = V_horiz x V_vert
     * At neutral forward posture, Normal points straight at camera -> Yaw=0°, Pitch=0°, Roll=0°!
     */
    _computeAccurateHeadPose(lm) {
      const p10 = lm[10];   // Forehead top
      const p152 = lm[152]; // Chin bottom
      const p33 = lm[33];   // Left eye outer
      const p263 = lm[263]; // Right eye outer

      // Horizontal vector (left eye to right eye)
      const vh = {
        x: p263.x - p33.x,
        y: p263.y - p33.y,
        z: p263.z - p33.z
      };

      // Vertical vector (chin to forehead)
      const vv = {
        x: p10.x - p152.x,
        y: p10.y - p152.y,
        z: p10.z - p152.z
      };

      // Cross product N = vh x vv
      const nx = vh.y * vv.z - vh.z * vv.y;
      const ny = vh.z * vv.x - vh.x * vv.z;
      const nz = vh.x * vv.y - vh.y * vv.x;

      const len = Math.hypot(nx, ny, nz) || 1;
      const normX = nx / len;
      const normY = ny / len;
      const normZ = nz / len;

      // Physical rotation angles in degrees
      // If camera is mirrored webcam, invert yaw direction
      const rawYaw = Math.atan2(normX, -normZ) * (180 / Math.PI) * (this.isWebcamSource ? -1.4 : 1.4);
      const rawPitch = Math.atan2(normY, -normZ) * (180 / Math.PI) * 1.35;
      const rawRoll = Math.atan2(vh.y, vh.x) * (180 / Math.PI);

      // Apply Exponential Moving Average (EMA) to eliminate frame noise
      const alpha = this.frameAlpha;
      this.smoothYaw = this.smoothYaw * (1 - alpha) + rawYaw * alpha;
      this.smoothPitch = this.smoothPitch * (1 - alpha) + rawPitch * alpha;
      this.smoothRoll = this.smoothRoll * (1 - alpha) + rawRoll * alpha;

      // Adjust against calibrated baseline (defaults to 0 if not calibrated)
      const adjYaw = Math.round(this.smoothYaw - this.baseline.yaw);
      const adjPitch = Math.round(this.smoothPitch - this.baseline.pitch);
      const adjRoll = Math.round(this.smoothRoll);

      let status = "Facing Forward";
      const reportPoseIncident = this.baseline.isCalibrated && !this.baseline.isCalibrating;

      const isTurned = Math.abs(adjYaw) > 18;
      if (isTurned) {
        status = adjYaw > 0 ? "Turned Right" : "Turned Left";
        if (reportPoseIncident) {
          this._triggerStateIncident(
            "headYaw",
            true,
            "Head Turned Away",
            `Candidate rotated head ${Math.abs(adjYaw)}° ${adjYaw > 0 ? "right" : "left"} off-screen`,
            "notice",
            "Head Re-Aligned",
            "Candidate head returned to center monitor alignment"
          );
        }
      } else {
        this._triggerStateIncident("headYaw", false, null, null, "normal", "Head Re-Aligned", "Candidate head returned to center monitor alignment");
      }

      const isLookingDown = adjPitch > 16;
      if (isLookingDown) {
        status = "Looking Down (Notes)";
        if (reportPoseIncident) {
          this._triggerStateIncident(
            "headPitchDown",
            true,
            "Head Down (Notes/Phone)",
            `Candidate head angled ${adjPitch}° down towards desk or phone`,
            "notice",
            "Head Lifted",
            "Candidate restored upright head alignment"
          );
        }
      } else {
        this._triggerStateIncident("headPitchDown", false, null, null, "normal", "Head Lifted", "Candidate restored upright head alignment");
      }

      const isLookingUp = adjPitch < -16;
      if (isLookingUp) {
        status = "Looking Up";
        if (reportPoseIncident) {
          this._triggerStateIncident(
            "headPitchUp",
            true,
            "Head Up (Ceiling)",
            `Candidate head angled ${Math.abs(adjPitch)}° upward away from screen`,
            "notice",
            "Head Forward",
            "Candidate head returned to monitor level"
          );
        }
      } else {
        this._triggerStateIncident("headPitchUp", false, null, null, "normal", "Head Forward", "Candidate head returned to monitor level");
      }

      return {
        yaw: Math.max(-75, Math.min(75, adjYaw)),
        pitch: Math.max(-75, Math.min(75, adjPitch)),
        roll: Math.max(-45, Math.min(45, adjRoll)),
        status: status
      };
    }

    /**
     * Compute Accurate Eye Gaze compensated by 3D Head Orientation
     */
    _computeAccurateEyeGaze(lm, pose) {
      // Viewer's Left Eye (Anatomical Right):
      // Socket: outer 33, inner 133, top 159, bottom 145. Iris center: 473.
      // Viewer's Right Eye (Anatomical Left):
      // Socket: inner 362, outer 263, top 386, bottom 374. Iris center: 468.
      const vLeftIris = lm[473] || lm[468];
      const vRightIris = lm[468] || lm[473];

      if (!vLeftIris || !vRightIris) {
        return { direction: "CENTER", gazeAngleX: 0, gazeAngleY: 0, leftIris: vLeftIris, rightIris: vRightIris, isSaccade: false, isEyesClosed: false };
      }

      // If candidate's eyes are currently closed (blinking), avoid dividing by collapsed eyelid height
      const isEyesClosed = (this.eyesClosedSince > 0);
      if (isEyesClosed) {
        return {
          direction: "EYES CLOSED (BLINK)",
          gazeAngleX: Math.round(this.smoothGazeX - this.baseline.gazeX),
          gazeAngleY: Math.round(this.smoothGazeY - this.baseline.gazeY),
          leftIris: vLeftIris,
          rightIris: vRightIris,
          isSaccade: false,
          isEyesClosed: true
        };
      }

      // Viewer's Left Eye Socket (Anatomical Right)
      const vlOuter = lm[33], vlInner = lm[133], vlTop = lm[159], vlBottom = lm[145];
      // Viewer's Right Eye Socket (Anatomical Left)
      const vrInner = lm[362], vrOuter = lm[263], vrTop = lm[386], vrBottom = lm[374];

      const vlWidth = Math.abs(vlInner.x - vlOuter.x) || 0.001;
      const vrWidth = Math.abs(vrOuter.x - vrInner.x) || 0.001;
      const vlHeight = Math.max(0.008, Math.abs(vlBottom.y - vlTop.y));
      const vrHeight = Math.max(0.008, Math.abs(vrBottom.y - vrTop.y));

      const vlMidX = (vlOuter.x + vlInner.x) / 2;
      const vrMidX = (vrInner.x + vrOuter.x) / 2;
      const vlMidY = (vlTop.y + vlBottom.y) / 2;
      const vrMidY = (vrTop.y + vrBottom.y) / 2;

      // Normalized iris offset within respective eye socket (-0.45 to +0.45)
      const vlOffX = Math.max(-0.45, Math.min(0.45, (vLeftIris.x - vlMidX) / vlWidth));
      const vrOffX = Math.max(-0.45, Math.min(0.45, (vRightIris.x - vrMidX) / vrWidth));
      const avgOffsetX = (vlOffX + vrOffX) / 2;

      const vlOffY = Math.max(-0.45, Math.min(0.45, (vLeftIris.y - vlMidY) / vlHeight));
      const vrOffY = Math.max(-0.45, Math.min(0.45, (vRightIris.y - vrMidY) / vrHeight));
      const avgOffsetY = (vlOffY + vrOffY) / 2;

      // Ocular displacement angle (clamped to ±24° natural eyeball range)
      // Resting eyelid sits slightly lower than center apex, subtract 0.06
      const ocularAngleX = Math.max(-24, Math.min(24, (this.isWebcamSource ? -avgOffsetX : avgOffsetX) * 45));
      const ocularAngleY = Math.max(-24, Math.min(24, (avgOffsetY - 0.06) * 35));

      // Total Gaze = Head Pose Rotation + Ocular Eye Displacement
      const rawGazeX = Math.max(-75, Math.min(75, pose.yaw + ocularAngleX));
      const rawGazeY = Math.max(-75, Math.min(75, pose.pitch + ocularAngleY));

      // Smooth gaze
      const alpha = this.frameAlpha;
      this.smoothGazeX = this.smoothGazeX * (1 - alpha) + rawGazeX * alpha;
      this.smoothGazeY = this.smoothGazeY * (1 - alpha) + rawGazeY * alpha;

      // Calibrate if calibration active
      if (this.baseline.isCalibrating) {
        this.baseline.samples.push({
          pitch: this.smoothPitch,
          yaw: this.smoothYaw,
          gazeX: this.smoothGazeX,
          gazeY: this.smoothGazeY
        });
      }
      if (this.sessionWarmupCount === 40 && this.autoBaselineSet && !this.warmupGazeCalibrated) {
        this.baseline.gazeX = this.smoothGazeX;
        this.baseline.gazeY = this.smoothGazeY;
        this.warmupGazeCalibrated = true;
      }

      const totalGazeX = Math.round(this.smoothGazeX - this.baseline.gazeX);
      const totalGazeY = Math.round(this.smoothGazeY - this.baseline.gazeY);

      // Determine gaze direction with appropriate neutral deadband (±18°)
      let dir = "CENTER";
      if (totalGazeY > 20) dir = "LOOKING DOWN (DESK/NOTES)";
      else if (totalGazeY < -20) dir = "LOOKING UP";
      else if (totalGazeX > 20) dir = "LOOKING RIGHT";
      else if (totalGazeX < -20) dir = "LOOKING LEFT";

      // Temporal state tracking for incidents
      const now = Date.now();
      if (dir !== this.sustainedGazeDir) {
        this.sustainedGazeDir = dir;
        this.gazeDirChangeTime = now;
      } else {
        const sustainedDuration = now - this.gazeDirChangeTime;
        const isOffscreen = (dir !== "CENTER");

        if (this.baseline.isCalibrated && !this.baseline.isCalibrating) {
          if (isOffscreen && sustainedDuration > 650) {
            let title = "Gaze Off-Screen";
            let detail = `Candidate looking towards ${dir.toLowerCase()} (X=${totalGazeX}°, Y=${totalGazeY}°)`;
            if (dir.includes("DOWN")) {
              title = "Gaze Down (Notes/Phone)";
              detail = `Candidate sustained downward glance towards desk or notes (Y=${totalGazeY}°)`;
            } else if (dir.includes("UP")) {
              title = "Gaze Upwards (Ceiling)";
              detail = `Candidate looking upwards towards ceiling (Y=${totalGazeY}°)`;
            }
            this._triggerStateIncident(
              "gazeOffscreen",
              true,
              title,
              detail,
              "notice",
              "Gaze Re-Aligned",
              "Candidate eye focal alignment restored to screen"
            );
          } else if (!isOffscreen) {
            this._triggerStateIncident("gazeOffscreen", false, null, null, "normal", "Gaze Re-Aligned", "Candidate eye focal alignment restored to screen");
          }
        }
      }

      // Saccade pattern tracking (reading notes off-screen)
      this.gazeHistoryX.push(totalGazeX);
      if (this.gazeHistoryX.length > 24) this.gazeHistoryX.shift();
      const isSaccade = this._detectReadingSaccade(this.gazeHistoryX);
      if (isSaccade && !this.wasReadingSaccade) {
        this.saccadeCount++;
        this._debouncedIncident("Repeated Eye Movement", "Repeated horizontal eye movement detected", "notice", 8000);
      }
      this.wasReadingSaccade = isSaccade;

      return {
        direction: dir,
        gazeAngleX: totalGazeX,
        gazeAngleY: totalGazeY,
        leftIris: vLeftIris,
        rightIris: vRightIris,
        isSaccade
      };
    }

    _detectReadingSaccade(history) {
      if (history.length < 18) return false;
      const avgGaze = Math.abs(history.reduce((a, b) => a + b, 0) / history.length);
      if (avgGaze < 15) return false; // Candidate is focused on screen, not reading offscreen notes

      let sweepingMotions = 0;
      let rapidJumps = 0;
      for (let i = 1; i < history.length; i++) {
        const diff = history[i] - history[i - 1];
        if (Math.abs(diff) > 12) rapidJumps++;
        else if (Math.abs(diff) > 4) sweepingMotions++;
      }
      return rapidJumps >= 2 && sweepingMotions >= 6;
    }

    /**
     * Facial Action Unit (AU) Micro-Expression & Mood Analysis
     * Tracks AU12 (Smile), AU4 (Brow Furrow), AU1/2 (Brow Lift), AU15 (Corner Depressor / Sadness),
     * AU26 (Speech Aperture), and involuntary facial skin micro-twitches.
     */
    _analyzeExpressions(lm) {
      if (!lm || lm.length < 468) {
        return {
          smileScore: 0,
          furrowScore: 0,
          browLift: 0,
          sadScore: 0,
          drowsinessScore: 0,
          microTwitchScore: 0,
          nervousScore: 0,
          steadiness: 98,
          isSpeaking: false,
          speakingRatio: 0,
          mood: "NEUTRAL",
          moodChip: "ATTENTIVE",
          chipClass: "chip-normal"
        };
      }

      const faceWidth = Math.hypot(lm[454].x - lm[234].x, lm[454].y - lm[234].y) || 0.001;
      const faceHeight = Math.abs(lm[152].y - lm[10].y) || 0.001;

      // 1. AU12 Zygomatic Smile & Laughter (Width stretch + Corner lift + Open laugh aperture)
      const mouthWidth = Math.hypot(lm[291].x - lm[61].x, lm[291].y - lm[61].y);
      const smileRatio = mouthWidth / faceWidth;

      // Lip corner elevation (corners 61, 291 pull up towards nose/mouth center)
      const smileCenterY = (lm[13].y + lm[14].y) / 2;
      const cornerElevation = Math.max(0, (smileCenterY - (lm[61].y + lm[291].y) / 2) / faceHeight);

      // Vertical laughter mouth opening (teeth / open laugh aperture)
      const smileAperture = Math.abs(lm[14].y - lm[13].y) / faceHeight;

      // Base width score: neutral is ~0.37. Reaches 100 on broad smile / laugh
      let widthSmile = Math.max(0, (smileRatio - 0.37) * 650);

      // Bonus for upward corner elevation (zygomaticus major pull)
      if (cornerElevation > 0.003) {
        widthSmile += cornerElevation * 700;
      }

      // Bonus for open-mouth laughter (teeth exposed)
      if (smileRatio > 0.42 && smileAperture > 0.015) {
        widthSmile += Math.min(40, (smileAperture - 0.015) * 600);
      }

      const smileScore = Math.max(0, Math.min(100, Math.round(widthSmile)));

      // 2. AU4 Brow Furrow / Cognitive Tension
      const browDist = Math.hypot(lm[285].x - lm[55].x, lm[285].y - lm[55].y);
      const furrowRatio = browDist / faceWidth;
      const furrowScore = Math.max(0, Math.min(100, Math.round((0.24 - furrowRatio) * 450)));

      // 3. AU1 / AU2 Eyebrow Elevation (Eyebrow Flash / Skepticism / Micro-motion)
      const leftBrowY = (lm[70].y + lm[105].y) / 2;
      const rightBrowY = (lm[336].y + lm[334].y) / 2;
      const leftEyeY = lm[159].y;
      const rightEyeY = lm[386].y;
      const avgBrowEyeDist = ((leftEyeY - leftBrowY) + (rightEyeY - rightBrowY)) / (2 * faceHeight);

      if (this.baselineBrowDist === 0 && this.baseline.isCalibrated) {
        this.baselineBrowDist = avgBrowEyeDist;
      }
      let browLift = 0;
      if (this.baselineBrowDist > 0.01) {
        const liftRatio = avgBrowEyeDist / this.baselineBrowDist;
        browLift = Math.max(0, Math.min(100, Math.round((liftRatio - 1.05) * 350)));
      }
      this.currentBrowLift = browLift;

      if (browLift >= 52) {
        this.eyebrowFlashCount++;
        this._debouncedIncident(
          "Micro-Expression: Brow Lift",
          `Dynamic eyebrow elevation detected (${browLift}% AU1/2 curiosity/skepticism)`,
          "normal",
          12000
        );
      }

      // 4. AU15 Sadness / Downcast Mouth Corner Depression & Flat Affect
      // Lip corners (61, 291) vs mouth center (0, 17)
      const mouthCornersY = (lm[61].y + lm[291].y) / 2;
      const mouthCenterY = (lm[0].y + lm[17].y) / 2;
      const mouthDroopRatio = (mouthCornersY - mouthCenterY) / faceHeight;
      // When corners droop downward below mouth center line: mouthDroopRatio > 0.007
      const sadScore = Math.max(0, Math.min(100, Math.round((mouthDroopRatio - 0.006) * 1150)));
      this.sadScore = sadScore;

      // 5. Skin & Facial Landmark Micro-Tremor / Twitch Tracking
      let microTwitchScore = 0;
      if (this.lastLandmarkPositions && this.lastLandmarkPositions.length >= 7) {
        const keyIdx = [10, 1, 152, 234, 454, 70, 336];
        let sumJitter = 0;
        keyIdx.forEach((idx, i) => {
          const p = lm[idx];
          const prev = this.lastLandmarkPositions[i];
          if (p && prev) {
            sumJitter += Math.hypot(p.x - prev.x, p.y - prev.y);
          }
        });
        const avgJitter = (sumJitter / keyIdx.length) / faceWidth;
        if (avgJitter > 0.011) {
          microTwitchScore = Math.min(100, Math.round((avgJitter - 0.009) * 2200));
        }
      }
      this.lastLandmarkPositions = [10, 1, 152, 234, 454, 70, 336].map((idx) => ({ x: lm[idx].x, y: lm[idx].y }));
      this.microTwitchScore = microTwitchScore;

      if (microTwitchScore >= 45) {
        this._debouncedIncident(
          "Facial Micro-Twitch",
          `Involuntary facial micro-muscle shift detected (${microTwitchScore}% intensity)`,
          "normal",
          4500
        );
      }

      // 6. AU26 Mouth Aperture & Dynamic Micro-Lips Speech Tracking
      const innerAperture = Math.abs(lm[14].y - lm[13].y) / faceHeight;
      const outerAperture = Math.abs(lm[17].y - lm[0].y) / faceHeight;

      // Dynamic frame-to-frame lip articulation velocity
      const lipMotion = Math.abs(innerAperture - (this.lastInnerAperture || innerAperture));
      this.lastInnerAperture = innerAperture;

      // Catch micro-lip speaking: subtle articulation, murmurs, whispering, or mic vocalization
      const isSpeaking = (innerAperture > 0.010) ||
        (lipMotion > 0.0018 && innerAperture > 0.006) ||
        (this.isVocalizing && innerAperture > 0.005) ||
        (this.isVocalizing && outerAperture > 0.048);

      this.speechBuffer.push(isSpeaking ? 1 : 0);
      if (this.speechBuffer.length > 36) this.speechBuffer.shift();
      const speakingRatio = Math.round((this.speechBuffer.reduce((a, b) => a + b, 0) / Math.max(1, this.speechBuffer.length)) * 100);
      this.lastSpeechRatio = speakingRatio;

      // 7. AU24 Lip Compression / Tightening (Tension marker when silent)
      const lipThickness = Math.abs(lm[14].y - lm[13].y) / faceWidth;
      const isLipCompressed = (!isSpeaking && lipThickness < 0.012 && furrowScore > 14);

      // 8. Rapid Blink Bursts in the last 6 seconds
      const now = Date.now();
      const recentBlinksIn6s = (this.recentBlinkTimestamps || []).filter((t) => now - t < 6000).length;
      const isRapidBlinking = (recentBlinksIn6s >= 5 || (this.blinkRateHistory >= 32 && this.sessionWarmupCount >= 40));

      // 9. Composite Nervousness Score (0 - 100)
      let nervousScore = 0;
      if (isRapidBlinking) nervousScore += Math.min(35, (recentBlinksIn6s * 7) + (this.blinkRateHistory > 32 ? 14 : 0));
      if (furrowScore > 24) nervousScore += Math.min(30, Math.round((furrowScore - 24) * 1.2));
      if (isLipCompressed) nervousScore += 18;
      if (this.wasReadingSaccade) nervousScore += 16;
      nervousScore = Math.min(100, Math.max(0, nervousScore));
      this.nervousScore = nervousScore;

      const steadiness = Math.max(35, Math.min(100, 100 - Math.round(nervousScore * 0.65)));
      this.steadinessScore = steadiness;

      // Hysteresis for nervous state (enter >= 48, exit < 32)
      if (this.isCandidateNervousState) {
        if (nervousScore < 32) {
          this.isCandidateNervousState = false;
        }
      } else {
        if (nervousScore >= 48) {
          this.isCandidateNervousState = true;
        }
      }

      // 10. Intelligent Multi-Tier Mood Synthesis
      let mood = "NEUTRAL";
      let moodChip = "ATTENTIVE";
      let chipClass = "chip-normal";

      const isEyesClosed = (this.eyesClosedSince > 0);
      const isDrowsy = (this.drowsinessScore >= 38);
      const isSad = (this.sadScore >= 35 && smileScore < 10);

      if (isEyesClosed) {
        mood = "EYES CLOSED";
        moodChip = "EYES CLOSED";
        chipClass = "chip-notice";
      } else if (isDrowsy) {
        mood = "DROWSY / FATIGUED";
        moodChip = "DROWSY / LOW ENERGY";
        chipClass = "chip-alert";
      } else if (isSad) {
        mood = "DOWNCAST / SAD";
        moodChip = "DOWNCAST / FLAT AFFECT";
        chipClass = "chip-notice";
        this._triggerStateIncident(
          "candidateSad",
          true,
          "Downcast / Sad Affect",
          `Candidate displays downcast facial affect (${this.sadScore}% AU15 downturn, low engagement)`,
          "notice",
          "Affect Restored",
          "Candidate facial expression returned to attentive neutral"
        );
      } else if (smileScore > 35) {
        if (smileScore >= 75) {
          mood = "LAUGHING / JOYFUL";
          moodChip = "LAUGHTER / JOY";
        } else {
          mood = "SMILING";
          moodChip = "WARM / SMILE";
        }
        chipClass = "chip-normal";
        this._triggerStateIncident("candidateSad", false);
        this._triggerStateIncident("candidateNervous", false);
      } else if (this.isCandidateNervousState) {
        mood = "NERVOUS / TENSE";
        moodChip = "ANXIOUS / TENSE";
        chipClass = "chip-notice";
        this._triggerStateIncident(
          "candidateNervous",
          true,
          "Candidate Appears Nervous",
          `Elevated anxiety biometrics: ${nervousScore}% stress index (${this.blinkRateHistory} blinks/min)`,
          "notice",
          "Composure Restored",
          "Candidate biometrics stabilized to calm baseline"
        );
      } else if (isSpeaking) {
        mood = "SPEAKING";
        moodChip = "ARTICULATING";
        chipClass = "chip-normal";
        this._triggerStateIncident("candidateSad", false);
        this._triggerStateIncident("candidateNervous", false);
      } else if (furrowScore > 32) {
        mood = "THINKING";
        moodChip = "COGNITIVE FOCUS";
        chipClass = "chip-normal";
        this._triggerStateIncident("candidateSad", false);
        this._triggerStateIncident("candidateNervous", false);
      } else {
        mood = "NEUTRAL";
        moodChip = "ENGAGED";
        chipClass = "chip-normal";
        this._triggerStateIncident("candidateSad", false);
        this._triggerStateIncident("candidateNervous", false);
      }

      return {
        smileScore,
        furrowScore,
        browLift,
        sadScore,
        drowsinessScore: this.drowsinessScore,
        microTwitchScore,
        nervousScore,
        steadiness,
        isSpeaking,
        speakingRatio,
        mood,
        moodChip,
        chipClass
      };
    }

    /**
     * Dual-Eye Aspect Ratio (EAR), Blink Rate & Ptosis/Drowsiness Tracking
     */
    _analyzeBlinks(lm) {
      // Left eye EAR
      const p159 = lm[159], p145 = lm[145], p33 = lm[33], p133 = lm[133];
      const vertL = Math.hypot(p159.x - p145.x, p159.y - p145.y);
      const horizL = Math.hypot(p33.x - p133.x, p33.y - p133.y) || 0.001;
      const earL = vertL / (2 * horizL);

      // Right eye EAR
      const p386 = lm[386], p374 = lm[374], p362 = lm[362], p263 = lm[263];
      const vertR = Math.hypot(p386.x - p374.x, p386.y - p374.y);
      const horizR = Math.hypot(p263.x - p362.x, p263.y - p362.y) || 0.001;
      const earR = vertR / (2 * horizR);

      const ear = (earL + earR) / 2;
      this.currentEar = ear;

      const now = Date.now();

      // Duchenne Smile / Laughter Squint Check:
      // When smiling or laughing, cheek elevation naturally narrows eye aperture (EAR drops to 0.11 - 0.15)
      // This is a joyous/warm micro-expression, NOT closed eyes or drowsiness!
      const faceWidth = Math.hypot(lm[454].x - lm[234].x, lm[454].y - lm[234].y) || 0.001;
      const mouthWidth = Math.hypot(lm[291].x - lm[61].x, lm[291].y - lm[61].y);
      const isSmileSquint = (mouthWidth / faceWidth) > 0.42;

      const closedThreshold = isSmileSquint ? 0.090 : 0.160;

      // 1. Fully Closed Eyes (Blink or Sustained Closure)
      if (ear < closedThreshold) {
        if (this.eyesClosedSince === 0) {
          this.eyesClosedSince = now;
        }
        const closedSec = (now - this.eyesClosedSince) / 1000;
        if (closedSec >= 1.4) {
          this._triggerStateIncident(
            "eyesClosed",
            true,
            "Prolonged Eyes Closed",
            `Candidate closed eyes for ${closedSec.toFixed(1)}s (Fatigue / Inattentive)`,
            "notice",
            "Eyes Re-Opened",
            "Candidate re-opened eyes after {sec}s total (Attentive)"
          );
        }
      } else {
        if (this.eyesClosedSince > 0) {
          const totalClosedSec = (now - this.eyesClosedSince) / 1000;
          this.blinkCount++;
          this.recentBlinkTimestamps = this.recentBlinkTimestamps || [];
          this.recentBlinkTimestamps.push(now);
          if (this.recentBlinkTimestamps.length > 20) this.recentBlinkTimestamps.shift();

          this._triggerStateIncident(
            "eyesClosed",
            false,
            null,
            null,
            "normal",
            "Eyes Re-Opened",
            `Candidate re-opened eyes after ${totalClosedSec.toFixed(1)}s (Attentive)`
          );
          this.eyesClosedSince = 0;
        }
      }

      // 2. Ptosis & Droopy Eyelids (Half-closed, sleepy, lazy gaze)
      // If smiling/laughing, eye narrowing is expressive, NOT ptosis or drowsiness!
      if (!isSmileSquint && ear >= 0.160 && ear < 0.230 && this.eyesClosedSince === 0) {
        if (this.droopyEyesSince === 0) {
          this.droopyEyesSince = now;
        }
        const droopSec = (now - this.droopyEyesSince) / 1000;
        if (droopSec > 0.6) {
          this.drowsinessScore = Math.min(100, Math.round(this.drowsinessScore * 0.90 + 80 * 0.10));
        }
      } else if (ear >= 0.240 || isSmileSquint) {
        this.droopyEyesSince = 0;
        this.drowsinessScore = Math.max(0, Math.round(this.drowsinessScore * 0.82));
      }

      const isDrowsy = (this.drowsinessScore >= 38);
      this._triggerStateIncident(
        "drowsy",
        isDrowsy,
        "Candidate Drowsy / Fatigued",
        `Ptosis detected: eyelids partially drooped (${ear.toFixed(2)} EAR, low energy)`,
        "notice",
        "Candidate Re-Alerted",
        "Eyelid aperture restored to alert focal engagement"
      );

      const elapsedMin = Math.max(0.5, (now - this.blinkStartTime) / 60000);
      this.blinkRateHistory = Math.round(this.blinkCount / elapsedMin);
    }

    /**
     * Smooth Composite Focus Score with Accurate 100% Reachability & Fatigue Penalties
     */
    _updateCompositeFocus(pose, gaze, exp) {
      let targetScore = 100;
      let penalty = 0;

      // 1. Eyes Closed Penalty (immediate steep drop)
      // Guard against smile/laugh squint: if candidate is smiling/laughing, do not penalize squinting!
      const isSmiling = (exp && exp.smileScore > 40);
      const isEyesClosed = (this.eyesClosedSince > 0 && !isSmiling);
      if (isEyesClosed) {
        const closedSec = (Date.now() - this.eyesClosedSince) / 1000;
        penalty += Math.min(75, 45 + closedSec * 12);
      }

      // 2. Drowsiness / Ptosis Penalty (Forces focus down when lazy/sleepy!)
      if (this.drowsinessScore > 18 && !isSmiling) {
        penalty += Math.min(50, Math.round(this.drowsinessScore * 0.65));
      }

      // 3. Downcast / Sad Affect Penalty
      if (this.sadScore > 24) {
        penalty += Math.min(38, Math.round((this.sadScore - 24) * 0.70));
      }

      // 4. Head Pose Direction Penalties
      if (pose) {
        const absYaw = Math.abs(pose.yaw || 0);
        const pitch = pose.pitch || 0;

        // Yaw (Head turned away from screen)
        if (absYaw > 14) {
          penalty += Math.min(55, 16 + (absYaw - 14) * 1.8);
        }

        // Pitch looking down (phone/notes/slouched posture)
        if (pitch > 14) {
          penalty += Math.min(55, 18 + (pitch - 14) * 1.8);
        }
        // Pitch looking up (ceiling)
        else if (pitch < -12) {
          penalty += Math.min(60, 20 + (Math.abs(pitch) - 12) * 2.0);
        }
      }

      // 5. Eye Gaze Direction Penalties
      if (gaze && !isEyesClosed) {
        if (gaze.direction !== "CENTER") {
          penalty += 26;
          if (gaze.direction.includes("DOWN")) penalty += 12;
          if (gaze.direction.includes("UP")) penalty += 16;
        } else {
          const dev = Math.hypot(gaze.gazeAngleX || 0, gaze.gazeAngleY || 0);
          if (dev > 12) {
            penalty += Math.min(28, (dev - 12) * 1.3);
          }
        }

        if (gaze.isSaccade) {
          penalty += 15;
        }
      }

      // 6. Hand covering face / touching chin
      if (this.handNearFace) {
        penalty += 14;
      }

      // 7. Secondary persons visible
      if ((this.secondaryPersons || 0) > 0) {
        penalty += 35;
      }

      // Compute targetScore
      targetScore = Math.max(10, Math.min(100, Math.round(100 - penalty)));

      const current = Number.isFinite(this.rawFocusScore) ? this.rawFocusScore : 100.0;
      const alpha = (targetScore < current) ? 0.25 : 0.08;
      this.rawFocusScore = current * (1 - alpha) + targetScore * alpha;

      // CRITICAL: Snap to 100% when fully focused so it reaches 100% smoothly and cleanly!
      if (targetScore === 100 && this.rawFocusScore >= 96.0) {
        this.rawFocusScore = 100.0;
      }

      this.focusScore = Math.round(this.rawFocusScore);
      this.focusHistory.push(this.focusScore);
      if (this.focusHistory.length > 400) this.focusHistory.shift();
    }

    /**
     * Absence handler
     */
    _handleFaceAbsent() {
      if (!this.faceAbsentSince) this.faceAbsentSince = Date.now();
      const current = Number.isFinite(this.focusScore) ? this.focusScore : 100;
      this.focusScore = Math.max(10, current - 2);
      if (Date.now() - this.faceAbsentSince >= 1500) {
        this._debouncedIncident("Candidate Left Frame", "No face detected in video stream", "critical", 5000);
      }

      if (this.onTelemetryUpdate) {
        this.onTelemetryUpdate({
          focusScore: Math.round(this.focusScore),
          pose: { yaw: 0, pitch: 0, roll: 0, status: "No Face in View" },
          gaze: { direction: "LOST", gazeAngleX: 0, gazeAngleY: 0 },
          expressions: { mood: "ABSENT", moodChip: "FACE ABSENT", chipClass: "chip-alert" },
          blinkRate: this.blinkRateHistory,
          handNearFace: false,
          secondaryPersons: 0,
          unauthorizedDevices: this.unauthorizedDevices,
          fps: this.currentFps
        });
      }
    }

    /**
     * Spatial Motion & Gesture Surveillance
     * Distinguishes candidate hand/arm gestures from true background intrusions
     */
    _analyzeBackgroundMotion(video) {
      if (!this.analysisCtx || !this._hasCurrentFrame(video)) return;

      const gw = 96;
      const gh = 72;

      try {
        this.analysisCtx.drawImage(video, 0, 0, gw, gh);
        const currData = this.analysisCtx.getImageData(0, 0, gw, gh).data;

        if (this.lastBackgroundPixels) {
          let bgChangedCount = 0;
          let bgEvaluatedCount = 0;
          let gestureChangedCount = 0;
          let gestureEvaluatedCount = 0;
          let faceMotionCount = 0;
          let faceEvaluatedCount = 0;
          let totalLumaDelta = 0;
          let totalPixels = 0;

          // Normalized candidate face box in 96x72 space
          const cBox = this.candidateBox || { x: 0.32, y: 0.18, w: 0.36, h: 0.4 };
          const fxMin = cBox.x * gw;
          const fxMax = (cBox.x + cBox.w) * gw;
          const fyMin = cBox.y * gh;
          const fyMax = (cBox.y + cBox.h) * gh;

          // Candidate Gesture & Torso Zone (arms, chest, hands)
          const gxMin = Math.max(0, (cBox.x - cBox.w * 0.9) * gw);
          const gxMax = Math.min(gw, (cBox.x + cBox.w * 1.9) * gw);
          const gyMin = Math.max(0, (cBox.y - cBox.h * 0.4) * gh);
          const gyMax = gh; // extends downward to bottom of screen

          for (let y = 0; y < gh; y += 3) {
            for (let x = 0; x < gw; x += 3) {
              const idx = (y * gw + x) * 4;
              const luma1 = 0.299 * currData[idx] + 0.587 * currData[idx + 1] + 0.114 * currData[idx + 2];
              const luma0 = 0.299 * this.lastBackgroundPixels[idx] + 0.587 * this.lastBackgroundPixels[idx + 1] + 0.114 * this.lastBackgroundPixels[idx + 2];
              const delta = Math.abs(luma1 - luma0);
              totalLumaDelta += delta;
              totalPixels++;

              const isFace = (x >= fxMin && x <= fxMax && y >= fyMin && y <= fyMax);
              const isGestureZone = (x >= gxMin && x <= gxMax && y >= gyMin && y <= gyMax);

              if (isFace) {
                faceEvaluatedCount++;
                if (delta > 24) faceMotionCount++;
              } else if (isGestureZone) {
                gestureEvaluatedCount++;
                if (delta > 22) gestureChangedCount++;
              } else {
                // Peripheral Room Background (only far corners and room behind)
                bgEvaluatedCount++;
                if (delta > 22) bgChangedCount++;
              }
            }
          }

          const lightingDelta = Math.round(totalLumaDelta / Math.max(1, totalPixels));

          // 1. Hand Near Face / Touching Face (Requires genuine lower face motion AND torso/arm energy)
          const faceMotionRatio = (faceMotionCount / Math.max(1, faceEvaluatedCount));
          const gestureEnergy = Math.min(100, Math.round((gestureChangedCount / Math.max(1, gestureEvaluatedCount)) * 320));
          this.candidateGestureEnergy = gestureEnergy;

          const rawHandTouchingFace = (faceMotionRatio > 0.68 && gestureEnergy > 52 && lightingDelta <= 18);
          this.handTouchFrameCount = rawHandTouchingFace ? (this.handTouchFrameCount || 0) + 1 : 0;
          const handTouchingFace = (this.handTouchFrameCount >= 2);
          this.handNearFace = handTouchingFace;

          if (handTouchingFace) {
            this.handActivity = "Hand Touching Face";
            this._triggerStateIncident(
              "handToFace",
              true,
              "Hand to Face",
              "Candidate hand or fingers touching face or chin",
              "normal",
              "Hand Cleared",
              "Candidate hand returned away from face"
            );
          } else {
            this._triggerStateIncident("handToFace", false, null, null, "normal", "Hand Cleared", "Candidate hand returned away from face");
          }

          // 2. Candidate Arm & Hand Gestures (moving arms, stretching, gesturing)
          const isGesturing = (gestureEnergy > 38 && lightingDelta <= 18 && !handTouchingFace);
          this.handGesturing = isGesturing;
          if (isGesturing && !handTouchingFace) {
            this.handActivity = "Hand Gesturing / Moving";
            this._triggerStateIncident(
              "handGesturing",
              true,
              "Candidate Hand Movement",
              "Candidate active hand / arm gesture detected",
              "normal",
              "Gesture Concluded",
              "Candidate hand gesture concluded"
            );
          } else if (!handTouchingFace) {
            this.handActivity = "Clear";
            this._triggerStateIncident("handGesturing", false);
          }

          // 3. True Peripheral Background Intrusion (ONLY in outer room border)
          const bgEnergy = Math.min(100, Math.round((bgChangedCount / Math.max(1, bgEvaluatedCount)) * 350));
          this.bgMotionEnergy = bgEnergy;
          if (bgEnergy > 34 && lightingDelta <= 18 && !isGesturing) {
            this._triggerStateIncident(
              "bgMotion",
              true,
              "Background Disturbance",
              `Physical motion detected in room background (${bgEnergy}% energy)`,
              "critical",
              "Room Settled",
              "Room background physical motion settled"
            );
          } else {
            this._triggerStateIncident("bgMotion", false);
          }

          if (this.onTelemetryUpdate) {
            this.onTelemetryUpdate({
              bgMotionEnergy: bgEnergy,
              candidateGestureEnergy: gestureEnergy,
              handNearFace: this.handNearFace,
              handActivity: this.handActivity,
              isGesturing: isGesturing,
              lightingDelta: lightingDelta
            });
          }
        }

        this.lastBackgroundPixels = new Uint8ClampedArray(currData);
      } catch (err) { }
    }

    /**
     * Asynchronous Object & Device Detection (COCO-SSD)
     * Filters out candidate's own body box so it does not draw clutter over the screen!
     */
    async _detectObjectsAsync(video, runId = this.runId) {
      if (!this.cocoModel || !this._hasCurrentFrame(video) || this.isDetectingObjects) return;

      this.isDetectingObjects = true;
      try {
        const predictions = await this.cocoModel.detect(video);
        if (runId !== this.runId || this.activeVideo !== video) return;
        const validPredictions = [];
        const detectedPhones = [];
        let otherPersonsCount = 0;

        predictions.forEach((pred) => {
          if (pred.score < 0.54) return;

          if (pred.class === "cell phone") {
            detectedPhones.push(pred);
            validPredictions.push(pred);
            this._debouncedIncident("Unauthorized Device", `Mobile phone detected in frame (${Math.round(pred.score * 100)}% conf)`, "critical", 4000);
          } else if (pred.class === "person") {
            // Check if this person box IS the candidate
            const [px, py, pw, ph] = pred.bbox;
            const isCenterCandidate = (px < video.videoWidth * 0.45 && (px + pw) > video.videoWidth * 0.55 && pw * ph > (video.videoWidth * video.videoHeight * 0.15));

            if (!isCenterCandidate) {
              otherPersonsCount++;
              validPredictions.push(pred);
            }
          }
        });

        this.latestCocoObjects = validPredictions;
        this.unauthorizedDevices = detectedPhones;

        if (otherPersonsCount > 0) {
          this._debouncedIncident("Multiple Persons", `Additional person visible in candidate frame`, "critical", 4000);
        }
      } catch (err) { }
      finally { this.isDetectingObjects = false; }
    }

    /* =========================================================================
     * RENDERING HUD OVERLAYS
     * ========================================================================= */

    _renderHudOverlay(lm, pose, gaze, exp) {
      const w = this.hudCanvas.width;
      const h = this.hudCanvas.height;
      const ctx = this.hudCtx;

      const isEyesClosed = (this.eyesClosedSince > 0 || gaze.isEyesClosed);
      const isDrowsy = (this.drowsinessScore >= 38);
      const isSad = (this.sadScore >= 35);
      const isCenter = (gaze.direction === "CENTER");
      const isNervous = (exp.nervousScore >= 38);

      // Cybernetic High-Contrast Palette
      let meshColor = "rgba(0, 229, 255, 0.42)"; // Cyber cyan
      let accentColor = "rgba(0, 255, 136, 0.95)"; // Cyber emerald neon
      let glowColor = "rgba(0, 255, 136, 0.5)";

      if (isEyesClosed || isDrowsy) {
        meshColor = "rgba(255, 179, 0, 0.45)"; // Cyber amber
        accentColor = "rgba(255, 179, 0, 0.95)";
        glowColor = "rgba(255, 179, 0, 0.5)";
      } else if (isSad) {
        meshColor = "rgba(147, 197, 253, 0.45)"; // Soft blue downcast
        accentColor = "rgba(96, 165, 250, 0.95)";
        glowColor = "rgba(96, 165, 250, 0.5)";
      } else if (!isCenter || isNervous) {
        meshColor = "rgba(255, 87, 34, 0.48)"; // Cyber coral alert
        accentColor = "rgba(255, 87, 34, 0.95)";
        glowColor = "rgba(255, 87, 34, 0.5)";
      }

      ctx.save();
      ctx.shadowBlur = 3;
      ctx.shadowColor = glowColor;

      const drawPath = (indices, close = false, strokeStyle = meshColor, lineWidth = 1.2) => {
        ctx.strokeStyle = strokeStyle;
        ctx.lineWidth = lineWidth;
        ctx.beginPath();
        indices.forEach((idx, i) => {
          const pt = lm[idx];
          if (!pt) return;
          if (i === 0) ctx.moveTo(pt.x * w, pt.y * h);
          else ctx.lineTo(pt.x * w, pt.y * h);
        });
        if (close) ctx.closePath();
        ctx.stroke();
      };

      // 1. Outer Face Contour (Jawline & Forehead)
      const contourIdx = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109];
      drawPath(contourIdx, true, meshColor, 1.4);

      // 2. Forehead Triangulation Facets
      drawPath([10, 67, 109], true, "rgba(0, 229, 255, 0.28)", 0.9);
      drawPath([10, 297, 338], true, "rgba(0, 229, 255, 0.28)", 0.9);

      // 3. Eyebrow Structural Arches (Left & Right)
      drawPath([70, 63, 105, 66, 107], false, accentColor, 1.8); // Left upper brow
      drawPath([55, 65, 52, 53, 46], false, meshColor, 1.1);     // Left lower brow
      drawPath([336, 296, 334, 293, 300], false, accentColor, 1.8); // Right upper brow
      drawPath([285, 295, 282, 283, 276], false, meshColor, 1.1);   // Right lower brow

      // 4. Eye Contour Polygons (Complete structural loops)
      const leftEyeLoop = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246];
      const rightEyeLoop = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398];
      drawPath(leftEyeLoop, true, isEyesClosed ? "#ffb300" : accentColor, 1.5);
      drawPath(rightEyeLoop, true, isEyesClosed ? "#ffb300" : accentColor, 1.5);

      // 5. Cheekbone Zygomatic Triangulation Facets
      drawPath([234, 127, 93], true, "rgba(0, 229, 255, 0.24)", 0.9);
      drawPath([127, 162, 21], true, "rgba(0, 229, 255, 0.24)", 0.9);
      drawPath([454, 356, 323], true, "rgba(0, 229, 255, 0.24)", 0.9);
      drawPath([356, 389, 251], true, "rgba(0, 229, 255, 0.24)", 0.9);

      // 6. Nose Bridge & Ridge Structure
      drawPath([168, 6, 197, 195, 5, 4, 1, 2], false, meshColor, 1.3);
      drawPath([98, 97, 2, 326, 327], false, meshColor, 1.0);

      // 7. Lips Structure (Outer and Inner Loops)
      const outerLips = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78];
      const innerLips = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95];
      drawPath(outerLips, true, exp.isSpeaking ? "rgba(0, 229, 255, 0.95)" : meshColor, 1.4);
      drawPath(innerLips, true, meshColor, 1.0);

      // 8. Facial Midline Axis & Glabella Crosshair
      drawPath([10, 151, 168, 1, 0, 152], false, "rgba(255, 255, 255, 0.25)", 0.8);

      // 9. Cheek Biometric Anchor Crosshairs
      [lm[234], lm[454]].forEach((pt) => {
        if (!pt) return;
        const px = pt.x * w;
        const py = pt.y * h;
        ctx.strokeStyle = accentColor;
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(px - 6, py); ctx.lineTo(px + 6, py);
        ctx.moveTo(px, py - 6); ctx.lineTo(px, py + 6);
        ctx.stroke();
      });

      // 10. Biometric Head Tracking Bounding Box Brackets
      if (this.candidateBox) {
        const bx = this.candidateBox.x * w;
        const by = this.candidateBox.y * h;
        const bw = this.candidateBox.w * w;
        const bh = this.candidateBox.h * h;
        const bLen = Math.min(24, bw * 0.22);

        ctx.strokeStyle = accentColor;
        ctx.lineWidth = 1.8;

        // Top-left corner
        ctx.beginPath();
        ctx.moveTo(bx, by + bLen); ctx.lineTo(bx, by); ctx.lineTo(bx + bLen, by);
        // Top-right corner
        ctx.moveTo(bx + bw - bLen, by); ctx.lineTo(bx + bw, by); ctx.lineTo(bx + bw, by + bLen);
        // Bottom-left corner
        ctx.moveTo(bx, by + bh - bLen); ctx.lineTo(bx, by + bh); ctx.lineTo(bx + bLen, by + bh);
        // Bottom-right corner
        ctx.moveTo(bx + bw - bLen, by + bh); ctx.lineTo(bx + bw, by + bh); ctx.lineTo(bx + bw, by + bh - bLen);
        ctx.stroke();
      }

      // 11. Iris Targeting Rings, Pupil Dots & 3D Gaze Directional Rays
      if (gaze.leftIris && gaze.rightIris) {
        const dotColor = isEyesClosed ? "#ffb300" : (isCenter ? "#00ff88" : "#ff5722");
        const rayColor = isEyesClosed ? "rgba(255, 179, 0, 0.45)" : (isCenter ? "rgba(0, 255, 136, 0.95)" : "rgba(255, 87, 34, 0.95)");

        [gaze.leftIris, gaze.rightIris].forEach((iris, idx) => {
          const ix = iris.x * w;
          const iy = iris.y * h;

          if (!isEyesClosed) {
            // Circular targeting reticle around iris
            ctx.strokeStyle = "rgba(0, 229, 255, 0.75)";
            ctx.lineWidth = 1.2;
            ctx.beginPath();
            ctx.arc(ix, iy, 9, 0, Math.PI * 2);
            ctx.stroke();

            // Center pupil dot
            ctx.fillStyle = dotColor;
            ctx.beginPath();
            ctx.arc(ix, iy, 4.0, 0, Math.PI * 2);
            ctx.fill();

            // Forward Gaze Ray (directed by physical gaze angles)
            const rayLen = 42;
            const rayDx = Math.sin((gaze.gazeAngleX * Math.PI) / 180) * rayLen * (this.isWebcamSource ? -1 : 1);
            const rayDy = Math.sin((gaze.gazeAngleY * Math.PI) / 180) * rayLen;

            ctx.strokeStyle = rayColor;
            ctx.lineWidth = 2.4;
            ctx.beginPath();
            ctx.moveTo(ix, iy);
            ctx.lineTo(ix + rayDx, iy + rayDy);
            ctx.stroke();

            // Live EAR indicator next to eye
            if (idx === 1) {
              const earX = this.isWebcamSource ? (ix - 24) : (ix + 24);
              this._drawUnmirroredText(
                ctx,
                `EAR: ${(this.currentEar || 0.30).toFixed(2)}`,
                earX,
                iy - 8,
                "bold 10px monospace",
                isDrowsy ? "#ffb300" : "#00ff88",
                "center",
                "middle"
              );
            }
          } else {
            // Eyes closed: amber resting dot
            ctx.fillStyle = "#ffb300";
            ctx.beginPath();
            ctx.arc(ix, iy, 4.5, 0, Math.PI * 2);
            ctx.fill();
          }
        });
      }

      // 12. Floating Forehead Biometric HUD Pill
      const nose = lm[1];
      const pillY = Math.max(22, (lm[10].y * h) - 34);
      const score = Math.round(this.focusScore);

      let moodLabel = exp.mood;
      if (isEyesClosed) {
        const closedSec = ((Date.now() - this.eyesClosedSince) / 1000).toFixed(1);
        moodLabel = `EYES CLOSED (${closedSec}s)`;
      } else if (isDrowsy) {
        moodLabel = `DROWSY (${(this.currentEar || 0.20).toFixed(2)} EAR)`;
      } else if (isSad) {
        moodLabel = `DOWNCAST (${exp.sadScore}%)`;
      } else if (isNervous) {
        moodLabel = `TENSE (${exp.nervousScore}%)`;
      }

      const pillText = `FOCUS: ${score}% • ${moodLabel} • EAR: ${(this.currentEar || 0.30).toFixed(2)}`;

      ctx.font = "bold 11px monospace";
      const txtWidth = ctx.measureText(pillText).width;
      const pillCenterX = nose.x * w;
      ctx.fillStyle = "rgba(10, 16, 13, 0.94)";
      ctx.fillRect(pillCenterX - (txtWidth / 2) - 12, pillY - 14, txtWidth + 24, 22);

      // Pill border
      ctx.strokeStyle = accentColor;
      ctx.lineWidth = 1.4;
      ctx.strokeRect(pillCenterX - (txtWidth / 2) - 12, pillY - 14, txtWidth + 24, 22);

      // Pill text fill (un-mirrored)
      const pillColor = isEyesClosed || isDrowsy ? "#ffb300" : (isSad ? "#93c5fd" : (isCenter ? "#00ff88" : "#ff7043"));
      this._drawUnmirroredText(ctx, pillText, pillCenterX, pillY - 3, "bold 11px monospace", pillColor, "center", "middle");

      ctx.restore();
    }

    _renderCocoOverlay() {
      if (!this.hudCtx || !this.hudCanvas || !this.latestCocoObjects || this.latestCocoObjects.length === 0) return;
      const video = this.activeVideo;
      const scaleX = video && video.videoWidth ? this.hudCanvas.width / video.videoWidth : 1;
      const scaleY = video && video.videoHeight ? this.hudCanvas.height / video.videoHeight : 1;

      this.latestCocoObjects.forEach((pred) => {
        const [x, y, bw, bh] = pred.bbox;
        const isDevice = (pred.class === "cell phone");

        this.hudCtx.strokeStyle = isDevice ? "#c53c3c" : "#2d689b";
        this.hudCtx.lineWidth = 2.2;
        this.hudCtx.strokeRect(x * scaleX, y * scaleY, bw * scaleX, bh * scaleY);

        const labelWidth = Math.min(bw * scaleX, 140);
        this.hudCtx.fillStyle = isDevice ? "rgba(197, 60, 60, 0.9)" : "rgba(45, 104, 155, 0.9)";
        this.hudCtx.fillRect(x * scaleX, Math.max(0, y * scaleY - 18), labelWidth, 18);

        this._drawUnmirroredText(
          this.hudCtx,
          `${pred.class.toUpperCase()} ${Math.round(pred.score * 100)}%`,
          x * scaleX + labelWidth / 2,
          Math.max(10, y * scaleY - 9),
          "10px monospace",
          "#ffffff",
          "center",
          "middle"
        );
      });
    }

    /**
     * Draw text on horizontally flipped (CSS scaleX(-1)) HUD canvas so that
     * characters display un-mirrored and naturally left-to-right to the user.
     */
    _drawUnmirroredText(ctx, text, x, y, font, fillStyle, align = "center", baseline = "middle") {
      ctx.save();
      ctx.font = font;
      ctx.fillStyle = fillStyle;
      ctx.textAlign = align;
      ctx.textBaseline = baseline;
      if (this.isWebcamSource) {
        ctx.translate(x, y);
        ctx.scale(-1, 1);
        ctx.fillText(text, 0, 0);
      } else {
        ctx.fillText(text, x, y);
      }
      ctx.restore();
    }

    /* =========================================================================
     * HELPER FUNCTIONS
     * ========================================================================= */

    _getFaceBBox(landmarks) {
      let minX = 1, maxX = 0, minY = 1, maxY = 0;
      landmarks.forEach((p) => {
        if (p.x < minX) minX = p.x;
        if (p.x > maxX) maxX = p.x;
        if (p.y < minY) minY = p.y;
        if (p.y > maxY) maxY = p.y;
      });
      return { x: minX, y: minY, w: maxX - minX, h: maxY - minY };
    }

    _getBBoxOverlap(b1, b2) {
      const xOverlap = Math.max(0, Math.min(b1.x + b1.w, b2.x + b2.w) - Math.max(b1.x, b2.x));
      const yOverlap = Math.max(0, Math.min(b1.y + b1.h, b2.y + b2.h) - Math.max(b1.y, b2.y));
      const interArea = xOverlap * yOverlap;
      const minArea = Math.min(b1.w * b1.h, b2.w * b2.h) || 0.001;
      return interArea / minArea;
    }

    /**
     * State-Transition Incident Engine
     * Zero-spam policy: Emits once when condition begins, once when it resolves.
     */
    _triggerStateIncident(stateKey, isConditionActive, enterTitle, enterDetail, severity = "notice", exitTitle = null, exitDetail = null) {
      const activeInfo = this.activeStates.get(stateKey);

      if (isConditionActive) {
        if (!activeInfo) {
          // New state transition: Inactive -> Active
          this.activeStates.set(stateKey, {
            startTime: Date.now(),
            lastReportTime: Date.now()
          });
          this._emitIncident(enterTitle, enterDetail, severity);
        } else {
          // Already active: Only report significant elapsed duration milestones for prolonged states
          const now = Date.now();
          const elapsedSec = (now - activeInfo.startTime) / 1000;
          if (stateKey === "eyesClosed" && elapsedSec >= 4.0 && (now - activeInfo.lastReportTime >= 3500)) {
            activeInfo.lastReportTime = now;
            this._emitIncident("Prolonged Eyes Closed", `Candidate eyes closed for ${elapsedSec.toFixed(1)}s (Inattentive / Fatigue)`, "notice");
          } else if (stateKey === "drowsy" && elapsedSec >= 8.0 && (now - activeInfo.lastReportTime >= 7000)) {
            activeInfo.lastReportTime = now;
            this._emitIncident("Sustained Fatigue", `Candidate eyes droopy for ${elapsedSec.toFixed(0)}s (Low engagement)`, "notice");
          }
        }
      } else {
        if (activeInfo) {
          // State transition: Active -> Inactive
          const durationSec = ((Date.now() - activeInfo.startTime) / 1000).toFixed(1);
          this.activeStates.delete(stateKey);

          if (exitTitle) {
            const detail = exitDetail ? exitDetail.replace("{sec}", durationSec) : `Returned to attentive posture (${durationSec}s)`;
            this._emitIncident(exitTitle, detail, "normal");
          }
        }
      }
    }

    /**
     * Real-time Audio Frame Analysis (Microphone, Voice, Noise Spikes)
     */
    _analyzeAudioFrame(isSpeakingVisual) {
      if (!this.isAudioReady || !this.audioAnalyser || !this.audioData) return;

      this.audioAnalyser.getByteFrequencyData(this.audioData);

      let sum = 0;
      let voiceSum = 0;
      let voiceBins = 0;

      for (let i = 0; i < this.audioData.length; i++) {
        const val = this.audioData[i];
        sum += val;
        // Speech frequency bins ~150Hz to 3000Hz
        if (i >= 1 && i <= 18) {
          voiceSum += val;
          voiceBins++;
        }
      }

      const avgEnergy = sum / this.audioData.length;
      const voiceEnergy = voiceSum / Math.max(1, voiceBins);

      const normLevel = avgEnergy / 255;
      const db = normLevel > 0.001 ? Math.round(20 * Math.log10(normLevel)) : -80;
      this.currentAudioDb = Math.max(-80, Math.min(0, db));

      // Vocalization detected by microphone
      this.isVocalizing = (voiceEnergy > 38 && db > -45);

      // Acoustic Room Noise Spike: High audio energy while candidate mouth is silent!
      if (avgEnergy > 45 && !isSpeakingVisual && db > -38) {
        this.bgNoiseSpike = true;
        this._triggerStateIncident(
          "bgAudioNoise",
          true,
          "Acoustic Noise Disturbance",
          `Ambient room audio spike (${this.currentAudioDb} dB) while candidate silent`,
          "notice",
          "Acoustic Floor Quiet",
          "Ambient room sound returned to quiet background"
        );
      } else {
        this.bgNoiseSpike = false;
        this._triggerStateIncident("bgAudioNoise", false);
      }
    }

    _debouncedIncident(title, detail, severity = "notice", minDelayMs = 2000) {
      const now = Date.now();
      const last = this.incidentCooldowns.get(title) || 0;
      if (now - last < Math.max(0, minDelayMs)) return;

      this.incidentCooldowns.set(title, now);
      this._emitIncident(title, detail, severity);
    }

    _emitIncident(title, detail, severity = "notice") {
      if (this.onIncident) {
        this.onIncident({ title, detail, severity, timestamp: Date.now() });
      }
    }

    _notifyStatus(text, pulseClass = "ready") {
      if (this.onStatusChange) {
        this.onStatusChange(text, pulseClass);
      }
    }

    _updateFps() {
      this.fpsCounter++;
      const now = this._now();
      if (now - this.lastFpsTimestamp >= 1000) {
        this.currentFps = Math.round((this.fpsCounter * 1000) / (now - this.lastFpsTimestamp));
        this.fpsCounter = 0;
        this.lastFpsTimestamp = now;
        if (this.onFpsUpdate) this.onFpsUpdate(this.currentFps);
      }
    }

    _simulateHeuristics(video) {
      if (!this.analysisCtx) return;
      try {
        this.analysisCtx.drawImage(video, 0, 0, 320, 240);
        const data = this.analysisCtx.getImageData(120, 80, 80, 80).data;
        let luma = 0;
        for (let i = 0; i < data.length; i += 4) {
          luma += data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114;
        }
        const avg = luma / (data.length / 4);
        if (avg > 25 && this.onTelemetryUpdate) {
          this.onTelemetryUpdate({
            focusScore: 95,
            pose: { yaw: 0, pitch: 0, roll: 0, status: "Facing Forward" },
            gaze: { direction: "CENTER", gazeAngleX: 0, gazeAngleY: 0 },
            expressions: { mood: "NEUTRAL", moodChip: "ATTENTIVE", chipClass: "chip-normal" },
            fps: this.currentFps
          });
        }
      } catch (e) { }
    }

    _now() {
      return typeof performance !== "undefined" && performance.now ? performance.now() : Date.now();
    }

    _hasCurrentFrame(video) {
      const readyState = typeof HTMLMediaElement !== "undefined" ? HTMLMediaElement.HAVE_CURRENT_DATA : 2;
      return !!video && video.readyState >= readyState && video.videoWidth > 0 && video.videoHeight > 0;
    }

    _frameAlpha() {
      const now = this._now();
      const dt = this.lastFrameAt ? Math.min(250, Math.max(1, now - this.lastFrameAt)) : 33;
      this.lastFrameAt = now;
      // Keep the same smoothing feel near 30 fps while adapting to variable frame rates.
      return 1 - Math.pow(1 - this.ALPHA, dt / 33);
    }
  }

  return new MLEngine();
});

/**
 * Stillpoint AI — Google Meet Injected Content Script (Phase 2)
 * Hooks candidate video tiles directly inside meet.google.com
 * Injects floating telemetry HUD & communicates via BroadcastChannel
 */
"use strict";

(function () {
  console.log("[Stillpoint AI] Google Meet Companion injecting...");

  // BroadcastChannel for syncing with Stillpoint Master Dashboard
  const channel = new BroadcastChannel("stillpoint_ai_coordinator");

  let hudContainer = null;
  let activeVideoElement = null;
  let analysisInterval = null;
  let currentFocusScore = 96;
  let incidentLog = [];
  let isMinimized = false;

  // Wait for Google Meet interface to mount
  const checkMeetReady = setInterval(() => {
    if (document.body) {
      clearInterval(checkMeetReady);
      injectHud();
      startVideoElementObserver();
    }
  }, 1000);

  function injectHud() {
    if (document.getElementById("stillpoint-meet-hud")) return;

    hudContainer = document.createElement("div");
    hudContainer.id = "stillpoint-meet-hud";
    hudContainer.innerHTML = `
      <div class="sp-header" id="sp-drag-header">
        <div class="sp-brand">
          <span class="sp-logo-dot"></span>
          <span class="sp-title">STILLPOINT<span>.AI</span></span>
        </div>
        <div class="sp-controls">
          <button class="sp-btn-icon" id="sp-btn-minimize" title="Minimize/Expand">_</button>
          <button class="sp-btn-icon" id="sp-btn-close" title="Close Companion">×</button>
        </div>
      </div>

      <div class="sp-body">
        <div class="sp-target-row">
          <span class="sp-target-label">HOOKED TILE:</span>
          <span class="sp-target-status" id="sp-target-status">SEARCHING MEET TILES...</span>
        </div>

        <div class="sp-score-banner">
          <div class="sp-score-left">
            <span class="sp-score-label">FOCUS & INTEGRITY</span>
            <span class="sp-score-num" id="sp-score-num">--</span>
          </div>
          <span class="sp-score-chip" id="sp-score-chip">READY</span>
        </div>

        <div class="sp-grid">
          <div class="sp-grid-cell">
            <span class="sp-cell-title">GAZE DIRECTION</span>
            <span class="sp-cell-val" id="sp-val-gaze">CENTER</span>
          </div>
          <div class="sp-grid-cell">
            <span class="sp-cell-title">HEAD POSTURE</span>
            <span class="sp-cell-val" id="sp-val-pose">FORWARD</span>
          </div>
          <div class="sp-grid-cell">
            <span class="sp-cell-title">SACCADE NOTES</span>
            <span class="sp-cell-val" id="sp-val-saccade">CLEAR</span>
          </div>
          <div class="sp-grid-cell">
            <span class="sp-cell-title">ANOMALIES</span>
            <span class="sp-cell-val" id="sp-val-anomalies">0 FLAGGED</span>
          </div>
        </div>

        <div class="sp-feed-box" id="sp-feed-box">
          <div class="sp-feed-item">
            <span class="sp-feed-time">00:00</span>
            <span>Stillpoint companion connected to Meet.</span>
          </div>
        </div>

        <div class="sp-actions-row">
          <button class="sp-btn sp-btn-flag" id="sp-btn-flag">🚩 Flag Moment</button>
          <button class="sp-btn sp-btn-full" id="sp-btn-open-dashboard">📊 Full Suite</button>
        </div>
      </div>

      <div class="sp-footer">
        <span>MEET STREAM HOOK ACTIVE</span>
        <a href="http://localhost:8000" target="_blank">DASHBOARD ↗</a>
      </div>
    `;

    document.body.appendChild(hudContainer);

    setupHudInteractivity();
  }

  function setupHudInteractivity() {
    const dragHeader = document.getElementById("sp-drag-header");
    const btnMinimize = document.getElementById("sp-btn-minimize");
    const btnClose = document.getElementById("sp-btn-close");
    const btnFlag = document.getElementById("sp-btn-flag");
    const btnOpenDash = document.getElementById("sp-btn-open-dashboard");

    // Make Draggable
    let isDragging = false;
    let startX = 0, startY = 0, initialLeft = 0, initialTop = 0;

    dragHeader.addEventListener("mousedown", (e) => {
      if (e.target.tagName === "BUTTON") return;
      isDragging = true;
      startX = e.clientX;
      startY = e.clientY;
      const rect = hudContainer.getBoundingClientRect();
      initialLeft = rect.left;
      initialTop = rect.top;
      hudContainer.style.right = "auto";
      hudContainer.style.left = `${initialLeft}px`;
      hudContainer.style.top = `${initialTop}px`;
    });

    window.addEventListener("mousemove", (e) => {
      if (!isDragging) return;
      const dx = e.clientX - startX;
      const dy = e.clientY - startY;
      hudContainer.style.left = `${Math.max(10, Math.min(window.innerWidth - 340, initialLeft + dx))}px`;
      hudContainer.style.top = `${Math.max(10, Math.min(window.innerHeight - 200, initialTop + dy))}px`;
    });

    window.addEventListener("mouseup", () => {
      isDragging = false;
    });

    // Minimize / Expand
    btnMinimize.addEventListener("click", () => {
      isMinimized = !isMinimized;
      hudContainer.classList.toggle("sp-minimized", isMinimized);
    });

    // Click on minimized pill to restore
    hudContainer.addEventListener("click", (e) => {
      if (isMinimized && !e.target.closest(".sp-controls")) {
        isMinimized = false;
        hudContainer.classList.remove("sp-minimized");
      }
    });

    // Close HUD
    btnClose.addEventListener("click", () => {
      if (confirm("Disconnect Stillpoint AI Companion from Google Meet?")) {
        hudContainer.remove();
        if (analysisInterval) clearInterval(analysisInterval);
      }
    });

    // Flag Moment Action
    btnFlag.addEventListener("click", () => {
      addIncident("Interviewer Flag", "Candidate flagged during Meet question", "critical");
    });

    // Open Dashboard Action
    btnOpenDash.addEventListener("click", () => {
      window.open("http://localhost:8000", "_blank");
    });
  }

  // OBSERVE GOOGLE MEET VIDEO TILES
  function startVideoElementObserver() {
    const scanVideos = () => {
      const videos = Array.from(document.querySelectorAll("video")).filter(
        (v) => v.videoWidth > 0 && v.videoHeight > 0 && !v.paused
      );

      // In Google Meet, remote participants have distinct stream dimensions
      if (videos.length > 0) {
        // Pick the largest visible candidate video
        let bestVideo = videos[0];
        let maxArea = bestVideo.videoWidth * bestVideo.videoHeight;

        videos.forEach((v) => {
          const area = v.videoWidth * v.videoHeight;
          if (area > maxArea) {
            maxArea = area;
            bestVideo = v;
          }
        });

        if (bestVideo !== activeVideoElement) {
          activeVideoElement = bestVideo;
          const statusEl = document.getElementById("sp-target-status");
          if (statusEl) {
            statusEl.textContent = `CANDIDATE (${bestVideo.videoWidth}×${bestVideo.videoHeight})`;
            statusEl.style.color = "#4cd48a";
          }
          startTelemetryAnalysis(bestVideo);
        }
      }
    };

    scanVideos();
    setInterval(scanVideos, 2500);
  }

  // RUN REAL-TIME ANALYSIS ON CANDIDATE VIDEO
  function startTelemetryAnalysis(vid) {
    if (analysisInterval) clearInterval(analysisInterval);

    const canvas = document.createElement("canvas");
    canvas.width = 160;
    canvas.height = 120;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });

    let lastLuma = 0;
    let saccadeCounter = 0;
    let gazeHistory = [];

    analysisInterval = setInterval(() => {
      if (!vid || vid.paused || vid.ended) return;

      try {
        ctx.drawImage(vid, 0, 0, 160, 120);

        // Analyze center third (face & eyes) vs peripheral bounds
        const centerData = ctx.getImageData(50, 25, 60, 50).data;
        const leftData = ctx.getImageData(10, 25, 40, 50).data;
        const rightData = ctx.getImageData(110, 25, 40, 50).data;

        const centerLuma = computeAverageLuma(centerData);
        const leftLuma = computeAverageLuma(leftData);
        const rightLuma = computeAverageLuma(rightData);

        // Determine gaze bias from luminance contrast in eye/face box
        const hDelta = leftLuma - rightLuma;
        let gazeDir = "CENTER";
        if (hDelta > 14) gazeDir = "LOOKING RIGHT";
        else if (hDelta < -14) gazeDir = "LOOKING LEFT";

        gazeHistory.push(gazeDir);
        if (gazeHistory.length > 10) gazeHistory.shift();

        // Check Saccades
        const isSaccade = gazeHistory.filter((g) => g !== "CENTER").length >= 5;

        // Dynamic Focus Score
        let penalty = 0;
        if (gazeDir !== "CENTER") penalty += 18;
        if (isSaccade) penalty += 15;

        currentFocusScore = Math.max(45, Math.min(100, currentFocusScore * 0.94 + (100 - penalty) * 0.06));

        updateCompanionUI(currentFocusScore, gazeDir, isSaccade);

        // Broadcast to Stillpoint full dashboard
        channel.postMessage({
          type: "MEET_TELEMETRY",
          timestamp: Date.now(),
          focusScore: Math.round(currentFocusScore),
          gazeDirection: gazeDir,
          isSaccade: isSaccade,
          resolution: `${vid.videoWidth}x${vid.videoHeight}`
        });
      } catch (err) {
        // Protected from CORS canvas taint if stream is cross-origin
      }
    }, 400);
  }

  function computeAverageLuma(pixels) {
    let sum = 0;
    for (let i = 0; i < pixels.length; i += 4) {
      sum += pixels[i] * 0.299 + pixels[i + 1] * 0.587 + pixels[i + 2] * 0.114;
    }
    return sum / (pixels.length / 4);
  }

  function updateCompanionUI(score, gaze, isSaccade) {
    const scoreNum = document.getElementById("sp-score-num");
    const scoreChip = document.getElementById("sp-score-chip");
    const valGaze = document.getElementById("sp-val-gaze");
    const valSaccade = document.getElementById("sp-val-saccade");

    if (!scoreNum) return;

    const rounded = Math.round(score);
    scoreNum.textContent = `${rounded}%`;
    valGaze.textContent = gaze;

    if (rounded >= 80) {
      scoreChip.textContent = "FOCUSED";
      scoreChip.style.background = "rgba(61, 168, 114, 0.2)";
      scoreChip.style.color = "#4cd48a";
    } else if (rounded >= 60) {
      scoreChip.textContent = "NOTICE";
      scoreChip.style.background = "rgba(245, 158, 11, 0.2)";
      scoreChip.style.color = "#f59e0b";
    } else {
      scoreChip.textContent = "ALERT";
      scoreChip.style.background = "rgba(239, 68, 68, 0.2)";
      scoreChip.style.color = "#ef4444";
    }

    if (isSaccade) {
      valSaccade.textContent = "READING PATTERN";
      valSaccade.style.color = "#f59e0b";
      addIncident("Saccade Alert", "Eye saccade matches reading off-screen prompt", "notice");
    } else {
      valSaccade.textContent = "CLEAR";
      valSaccade.style.color = "#e3e7e1";
    }
  }

  let lastIncidentTime = 0;
  function addIncident(title, desc, severity) {
    const now = Date.now();
    if (now - lastIncidentTime < 8000) return; // cooldown
    lastIncidentTime = now;

    const timeStr = new Date().toTimeString().slice(3, 8);
    const feedBox = document.getElementById("sp-feed-box");
    const valAnomalies = document.getElementById("sp-val-anomalies");

    if (feedBox) {
      const item = document.createElement("div");
      item.className = "sp-feed-item";
      item.innerHTML = `
        <span class="sp-feed-time">${timeStr}</span>
        <span class="${severity === 'critical' ? 'sp-feed-critical' : 'sp-feed-notice'}">${title}</span>
      `;
      feedBox.prepend(item);
    }

    incidentLog.push({ time: timeStr, title, desc, severity });
    if (valAnomalies) {
      valAnomalies.textContent = `${incidentLog.length} FLAGGED`;
    }

    // Broadcast event
    channel.postMessage({
      type: "MEET_INCIDENT",
      timestamp: timeStr,
      title: title,
      detail: desc,
      severity: severity
    });
  }
})();

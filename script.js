import {
  FilesetResolver,
  HandLandmarker
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

const startBtn = document.getElementById("startBtn");
const resetBtn = document.getElementById("resetBtn");
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const cursorEl = document.getElementById("cursor");
const buttons = [...document.querySelectorAll(".target-btn")];
const toast = document.getElementById("toast");

const systemStatus = document.getElementById("systemStatus");
const cameraStatus = document.getElementById("cameraStatus");
const handStatus = document.getElementById("handStatus");
const gestureStatus = document.getElementById("gestureStatus");
const hoverStatus = document.getElementById("hoverStatus");
const selectionStatus = document.getElementById("selectionStatus");
const modePill = document.getElementById("modePill");

const dwellRange = document.getElementById("dwellRange");
const dwellValue = document.getElementById("dwellValue");
const sensitivityRange = document.getElementById("sensitivityRange");
const sensitivityValue = document.getElementById("sensitivityValue");
const graspRange = document.getElementById("graspRange");
const graspValue = document.getElementById("graspValue");

let handLandmarker = null;
let stream = null;
let animationFrameId = null;
let running = false;
let lastVideoTime = -1;
let lastSeenTime = 0;

const LOST_HAND_GRACE_MS = 200;
const CURSOR_MAX_STEP = 0.035;

const state = {
  cursorX: 0.5,
  cursorY: 0.52,
  smoothedHandX: 0.5,
  isGrabbing: false,
  hasAnchor: false,
  anchorHandX: 0.5,
  anchorCursorX: 0.5,
  hoveredIndex: null,
  hoverStartTime: 0,
  activatedIndex: null,
  lastSelectedLabel: "None",
  smoothedGrabScore: 2.0
};

function setStatus(el, text, className = "") {
  el.textContent = text;
  el.className = `value ${className}`.trim();
}

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("show");
  window.clearTimeout(showToast._timer);
  showToast._timer = window.setTimeout(() => {
    toast.classList.remove("show");
  }, 900);
}

function updateSliderLabels() {
  dwellValue.textContent = `${Number(dwellRange.value)} ms`;
  sensitivityValue.textContent = `${Number(sensitivityRange.value).toFixed(1)}×`;
  graspValue.textContent = Number(graspRange.value).toFixed(2);
}

dwellRange.addEventListener("input", updateSliderLabels);
sensitivityRange.addEventListener("input", updateSliderLabels);
graspRange.addEventListener("input", updateSliderLabels);
updateSliderLabels();

function resizeOverlay() {
  const rect = video.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  overlay.width = Math.round(rect.width * dpr);
  overlay.height = Math.round(rect.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

window.addEventListener("resize", resizeOverlay);

async function setupHandLandmarker() {
  try {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
    );

    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "./hand_landmarker.task"
      },
      runningMode: "VIDEO",
      numHands: 1,
      minHandDetectionConfidence: 0.4,
      minHandPresenceConfidence: 0.4,
      minTrackingConfidence: 0.4
    });

    setStatus(systemStatus, "Ready", "good");
  } catch (error) {
    console.error(error);
    setStatus(systemStatus, "Model load failed", "bad");
    alert(
      "Could not load the hand tracker. Make sure hand_landmarker.task is in the same folder as the site files and that you are serving the page over HTTP(S)."
    );
  }
}

function distance(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = (a.z || 0) - (b.z || 0);
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function palmCenter(landmarks) {
  const points = [landmarks[0], landmarks[5], landmarks[9], landmarks[13], landmarks[17]];
  const sum = points.reduce((acc, p) => ({
    x: acc.x + p.x,
    y: acc.y + p.y,
    z: acc.z + (p.z || 0)
  }), { x: 0, y: 0, z: 0 });

  return {
    x: sum.x / points.length,
    y: sum.y / points.length,
    z: sum.z / points.length
  };
}

function extendedFingerCount(landmarks) {
  const palm = palmCenter(landmarks);
  const palmWidth = Math.max(distance(landmarks[5], landmarks[17]), 0.0001);

  const fingerSets = [
    { tip: 8, pip: 6 },
    { tip: 12, pip: 10 },
    { tip: 16, pip: 14 },
    { tip: 20, pip: 18 }
  ];

  let count = 0;

  for (const finger of fingerSets) {
    const tip = landmarks[finger.tip];
    const pip = landmarks[finger.pip];

    const tipPalm = distance(tip, palm) / palmWidth;
    const pipPalm = distance(pip, palm) / palmWidth;

    if (tipPalm > pipPalm + 0.18) {
      count += 1;
    }
  }

  return count;
}

function closedHandScore(landmarks) {
  const palm = palmCenter(landmarks);
  const palmWidth = Math.max(distance(landmarks[5], landmarks[17]), 0.0001);
  const tips = [8, 12, 16, 20];

  const avgTipDistance = tips
    .map(i => distance(landmarks[i], palm) / palmWidth)
    .reduce((sum, v) => sum + v, 0) / tips.length;

  const extended = extendedFingerCount(landmarks);

  return avgTipDistance + extended * 0.35;
}

function drawLandmarks(landmarks, isGrabbing) {
  const w = overlay.clientWidth;
  const h = overlay.clientHeight;

  ctx.clearRect(0, 0, w, h);
  ctx.save();
  ctx.translate(w, 0);
  ctx.scale(-1, 1);

  const connections = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [5,9],[9,10],[10,11],[11,12],
    [9,13],[13,14],[14,15],[15,16],
    [13,17],[17,18],[18,19],[19,20],
    [0,17]
  ];

  ctx.lineWidth = 2;
  ctx.strokeStyle = isGrabbing ? "#22c55e" : "#38bdf8";
  ctx.fillStyle = isGrabbing ? "#bbf7d0" : "#e0f2fe";

  for (const [a, b] of connections) {
    ctx.beginPath();
    ctx.moveTo(landmarks[a].x * w, landmarks[a].y * h);
    ctx.lineTo(landmarks[b].x * w, landmarks[b].y * h);
    ctx.stroke();
  }

  for (const point of landmarks) {
    ctx.beginPath();
    ctx.arc(point.x * w, point.y * h, 4, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.restore();
}

function clearOverlay() {
  ctx.clearRect(0, 0, overlay.clientWidth, overlay.clientHeight);
}

function updateCursorUi() {
  const prototypeRect = cursorEl.parentElement.getBoundingClientRect();
  const x = state.cursorX * prototypeRect.width;
  const y = state.cursorY * prototypeRect.height;
  cursorEl.style.left = `${x}px`;
  cursorEl.style.top = `${y}px`;
  cursorEl.classList.toggle("grabbed", state.isGrabbing);
}

function clearButtonStates() {
  buttons.forEach((btn) => btn.classList.remove("hovered", "activated"));
}

function updateHover(nowMs) {
  clearButtonStates();

  const cursorRect = cursorEl.getBoundingClientRect();
  const cursorCenterX = cursorRect.left + cursorRect.width / 2;
  const cursorCenterY = cursorRect.top + cursorRect.height / 2;

  let hovered = null;
  buttons.forEach((btn, index) => {
    const rect = btn.getBoundingClientRect();
    const inside =
      cursorCenterX >= rect.left &&
      cursorCenterX <= rect.right &&
      cursorCenterY >= rect.top &&
      cursorCenterY <= rect.bottom;
    if (inside) hovered = index;
  });

  state.hoveredIndex = hovered;

  if (hovered === null || !state.isGrabbing) {
    state.hoverStartTime = 0;
    hoverStatus.textContent = "None";
    return;
  }

  const btn = buttons[hovered];
  btn.classList.add("hovered");
  hoverStatus.textContent = btn.textContent;

  if (state.activatedIndex !== hovered) {
    if (state.hoverStartTime === 0) {
      state.hoverStartTime = nowMs;
    }

    const dwellMs = Number(dwellRange.value);
    const heldMs = nowMs - state.hoverStartTime;

    if (heldMs >= dwellMs) {
      state.activatedIndex = hovered;
      btn.classList.remove("hovered");
      btn.classList.add("activated");
      state.lastSelectedLabel = btn.textContent;
      selectionStatus.textContent = state.lastSelectedLabel;
      showToast(`${btn.textContent} selected`);
    }
  }

  if (state.activatedIndex !== null) {
    buttons[state.activatedIndex]?.classList.add("activated");
  }
}

function resetPrototype() {
  state.cursorX = 0.5;
  state.cursorY = 0.52;
  state.smoothedHandX = 0.5;
  state.isGrabbing = false;
  state.hasAnchor = false;
  state.anchorHandX = 0.5;
  state.anchorCursorX = 0.5;
  state.hoveredIndex = null;
  state.hoverStartTime = 0;
  state.activatedIndex = null;
  state.lastSelectedLabel = "None";
  state.smoothedGrabScore = 2.0;
  lastSeenTime = 0;
  selectionStatus.textContent = "None";
  hoverStatus.textContent = "None";
  setStatus(handStatus, "No", "bad");
  setStatus(gestureStatus, "Open / unknown", "warn");
  modePill.textContent = "Idle";
  clearButtonStates();
  updateCursorUi();
}

resetBtn.addEventListener("click", resetPrototype);

function updateFromLandmarks(landmarks, nowMs) {
  const rawX = landmarks[9].x;
  state.smoothedHandX = state.smoothedHandX * 0.85 + rawX * 0.15;

  const rawGrabScore = closedHandScore(landmarks);
  state.smoothedGrabScore = state.smoothedGrabScore * 0.8 + rawGrabScore * 0.2;

  const closeThreshold = Number(graspRange.value);
  const openThreshold = closeThreshold + 0.25;

  if (!state.isGrabbing) {
    if (state.smoothedGrabScore < closeThreshold) {
      state.isGrabbing = true;
      state.hasAnchor = false;
    }
  } else {
    if (state.smoothedGrabScore > openThreshold) {
      state.isGrabbing = false;
      state.hasAnchor = false;
      state.hoverStartTime = 0;
    }
  }

  if (state.isGrabbing) {
    if (!state.hasAnchor) {
      state.hasAnchor = true;
      state.anchorHandX = state.smoothedHandX;
      state.anchorCursorX = state.cursorX;
    }

    const movement = (state.anchorHandX - state.smoothedHandX) * Number(sensitivityRange.value);
    const targetX = Math.min(0.95, Math.max(0.05, state.anchorCursorX + movement));
    const delta = targetX - state.cursorX;
    state.cursorX += Math.max(-CURSOR_MAX_STEP, Math.min(CURSOR_MAX_STEP, delta));

    modePill.textContent = "Cursor grasped";
    setStatus(gestureStatus, `Closed hand (${state.smoothedGrabScore.toFixed(2)})`, "good");
  } else {
    modePill.textContent = "Idle";
    setStatus(gestureStatus, `Open hand (${state.smoothedGrabScore.toFixed(2)})`, "warn");
  }

  updateCursorUi();
  updateHover(nowMs);
  drawLandmarks(landmarks, state.isGrabbing);
}

async function startCamera() {
  if (!handLandmarker) {
    alert("The hand tracking model is not ready yet.");
    return;
  }

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: "user",
        frameRate: { ideal: 30, max: 60 }
      },
      audio: false
    });

    video.srcObject = stream;
    await video.play();
    resizeOverlay();

    running = true;
    setStatus(cameraStatus, "Running", "good");
    startBtn.disabled = true;
    startBtn.textContent = "Camera Running";
    renderLoop();
  } catch (error) {
    console.error(error);
    setStatus(cameraStatus, "Permission denied", "bad");
    alert("Could not start the camera. Check webcam permission and make sure the page is running on HTTPS or localhost.");
  }
}

function stopCamera() {
  running = false;
  if (animationFrameId) cancelAnimationFrame(animationFrameId);
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }
  video.srcObject = null;
  startBtn.disabled = false;
  startBtn.textContent = "Start Camera";
  setStatus(cameraStatus, "Stopped", "warn");
  clearOverlay();
}

startBtn.addEventListener("click", startCamera);
window.addEventListener("beforeunload", stopCamera);

function renderLoop() {
  if (!running) return;

  const nowMs = performance.now();

  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    const result = handLandmarker.detectForVideo(video, nowMs);
    lastVideoTime = video.currentTime;

    if (result.landmarks && result.landmarks.length > 0) {
      const landmarks = result.landmarks[0];
      lastSeenTime = nowMs;
      setStatus(handStatus, "Yes", "good");
      updateFromLandmarks(landmarks, nowMs);
    } else {
      const recentlySeen = nowMs - lastSeenTime < LOST_HAND_GRACE_MS;
      clearOverlay();

      if (recentlySeen) {
        setStatus(handStatus, "Recently lost", "warn");
        modePill.textContent = state.isGrabbing ? "Cursor grasped" : "Idle";
        hoverStatus.textContent = state.hoveredIndex === null ? "None" : buttons[state.hoveredIndex]?.textContent || "None";
      } else {
        setStatus(handStatus, "No", "bad");
        setStatus(gestureStatus, "Open / unknown", "warn");
        state.isGrabbing = false;
        state.hasAnchor = false;
        state.hoverStartTime = 0;
        modePill.textContent = "Idle";
        hoverStatus.textContent = "None";
        clearButtonStates();
        if (state.activatedIndex !== null) {
          buttons[state.activatedIndex]?.classList.add("activated");
        }
      }

      updateCursorUi();
    }
  }

  animationFrameId = requestAnimationFrame(renderLoop);
}

await setupHandLandmarker();
updateCursorUi();
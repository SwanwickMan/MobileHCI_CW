import {
  FilesetResolver,
  HandLandmarker
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

const startBtn = document.getElementById("startBtn");
const video = document.getElementById("video");
const cursorEl = document.getElementById("cursor");
const buttons = [...document.querySelectorAll(".target-btn")];

const systemStatus = document.getElementById("systemStatus");

let handLandmarker = null;
let stream = null;
let animationFrameId = null;
let running = false;
let lastVideoTime = -1;
let lastSeenTime = 0;

const LOST_HAND_GRACE_MS = 200;
const GESTURE_ON_MS = 90;
const GESTURE_OFF_MS = 140;

const state = {
  cursorX: 0.5,
  smoothedHandX: 0.5,
  isGrabbing: false,
  hasAnchor: false,
  anchorHandX: 0.5,
  anchorCursorX: 0.5,
  pendingSelectionIndex: null,
  wasGrabbingLastFrame: false,
  activatedIndex: null,
  gestureSeenSince: 0,
  gestureLostSince: 0
};

function setStatus(el, text, className = "") {
  el.textContent = text;
  el.className = `value ${className}`.trim();
}

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
  const sum = points.reduce(
    (acc, p) => ({
      x: acc.x + p.x,
      y: acc.y + p.y,
      z: acc.z + (p.z || 0)
    }),
    { x: 0, y: 0, z: 0 }
  );

  return {
    x: sum.x / points.length,
    y: sum.y / points.length,
    z: sum.z / points.length
  };
}

function isFingerExtended(landmarks, tipIndex, pipIndex) {
  const palm = palmCenter(landmarks);
  const palmWidth = Math.max(distance(landmarks[5], landmarks[17]), 0.0001);

  const tipPalm = distance(landmarks[tipIndex], palm) / palmWidth;
  const pipPalm = distance(landmarks[pipIndex], palm) / palmWidth;

  return tipPalm > pipPalm + 0.18;
}

function isFingerGun(landmarks) {
  const indexExtended = isFingerExtended(landmarks, 8, 6);
  const middleExtended = isFingerExtended(landmarks, 12, 10);
  const ringExtended = isFingerExtended(landmarks, 16, 14);
  const pinkyExtended = isFingerExtended(landmarks, 20, 18);

  return indexExtended && middleExtended && !ringExtended && !pinkyExtended;
}

function updateCursorUi() {
  const prototypeRect = cursorEl.parentElement.getBoundingClientRect();
  const x = state.cursorX * prototypeRect.width;
  cursorEl.style.left = `${x}px`;
  cursorEl.classList.toggle("grabbed", state.isGrabbing);
}

function updateHover() {
  buttons.forEach((btn) => btn.classList.remove("hovered"));

  if (state.activatedIndex !== null) {
    buttons[state.activatedIndex]?.classList.add("activated");
  }

  if (!state.isGrabbing) {
    return;
  }

  const clampedX = Math.min(0.9999, Math.max(0, state.cursorX));
  const hovered = Math.floor(clampedX * buttons.length);

  state.pendingSelectionIndex = hovered;

  const btn = buttons[hovered];
  if (!btn) {
    return;
  }

  btn.classList.add("hovered");
}

function commitSelectionOnRelease() {
  const justReleased = state.wasGrabbingLastFrame && !state.isGrabbing;

  if (!justReleased) {
    return;
  }

  const selectedIndex = state.pendingSelectionIndex;

  if (selectedIndex === null || !buttons[selectedIndex]) {
    state.pendingSelectionIndex = null;
    return;
  }

  if (state.activatedIndex !== null) {
    buttons[state.activatedIndex]?.classList.remove("activated");
  }

  state.activatedIndex = selectedIndex;
  buttons[selectedIndex].classList.add("activated");

  state.pendingSelectionIndex = null;
}

function updateFromLandmarks(landmarks, nowMs) {
  const rawX = landmarks[9].x;
  state.smoothedHandX = state.smoothedHandX * 0.85 + rawX * 0.15;

  const fingerGun = isFingerGun(landmarks);

  if (fingerGun) {
    state.gestureLostSince = 0;
    if (state.gestureSeenSince === 0) {
      state.gestureSeenSince = nowMs;
    }
  } else {
    state.gestureSeenSince = 0;
    if (state.gestureLostSince === 0) {
      state.gestureLostSince = nowMs;
    }
  }

  if (!state.isGrabbing) {
    if (state.gestureSeenSince && nowMs - state.gestureSeenSince >= GESTURE_ON_MS) {
      state.isGrabbing = true;
      state.hasAnchor = false;
    }
  } else {
    if (state.gestureLostSince && nowMs - state.gestureLostSince >= GESTURE_OFF_MS) {
      state.isGrabbing = false;
      state.hasAnchor = false;
    }
  }

  if (state.isGrabbing) {
    if (!state.hasAnchor) {
      state.hasAnchor = true;
      state.anchorHandX = state.smoothedHandX;
      state.anchorCursorX = state.cursorX;
    }

    const movement = (state.anchorHandX - state.smoothedHandX) * 1.8;
    const targetX = Math.min(0.95, Math.max(0.05, state.anchorCursorX + movement));
    const delta = targetX - state.cursorX;
    state.cursorX += delta;

  }

  updateCursorUi();
  commitSelectionOnRelease();
  updateHover();
  state.wasGrabbingLastFrame = state.isGrabbing;
}

async function startCamera() {
  if (!handLandmarker) {
    alert("The hand tracking model is not ready yet.");
    return;
  }

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: "user",
        frameRate: { ideal: 24, max: 30 }
      },
      audio: false
    });

    video.srcObject = stream;
    await video.play();

    running = true;
    startBtn.disabled = true;
    startBtn.textContent = "Camera Running";
    renderLoop();
  } catch (error) {
    console.error(error);
    alert("Could not start the camera. Check webcam permission and make sure the page is running on HTTPS or localhost.");
  }
}

startBtn.addEventListener("click", startCamera);

function renderLoop() {
  if (!running) return;

  const nowMs = performance.now();

  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    const result = handLandmarker.detectForVideo(video, nowMs);
    lastVideoTime = video.currentTime;

    if (result.landmarks && result.landmarks.length > 0) {
      const landmarks = result.landmarks[0];
      lastSeenTime = nowMs;
      updateFromLandmarks(landmarks, nowMs);
    } else {
      const recentlySeen = nowMs - lastSeenTime < LOST_HAND_GRACE_MS;

      if (!recentlySeen) {
        state.isGrabbing = false;
        state.hasAnchor = false;
        state.pendingSelectionIndex = null;
        state.gestureSeenSince = 0;
        state.gestureLostSince = 0;

        buttons.forEach((btn) => btn.classList.remove("hovered"));

        if (state.activatedIndex !== null) {
          buttons[state.activatedIndex]?.classList.add("activated");
        }
      }

      updateCursorUi();
      state.wasGrabbingLastFrame = state.isGrabbing;
    }
  }

  animationFrameId = requestAnimationFrame(renderLoop);
}

await setupHandLandmarker();
updateCursorUi();
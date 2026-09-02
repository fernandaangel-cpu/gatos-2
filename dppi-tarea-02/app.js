import {
  HandLandmarker,
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// ---- meme mapping (cartas) -------------------------------------------
const GESTURE_MEMES = {
  muerte: ["cartas/muerte.png"],
  sol: ["cartas/sol.jpeg"],
  mago: ["cartas/Mago.jpeg"],
  amantes: ["cartas/Amantes.jpeg"],
  diablo: ["cartas/Diablo.jpeg"],
  elLoco: ["cartas/Elloco.jpeg"],
  emperador: ["cartas/emperador.jpeg"],
};

// how many consecutive frames a gesture must hold before we switch to it
const STABLE_FRAMES_REQUIRED = 5;
// if no hand / no gesture is seen for this long, fall back to default (emperador)
const DEFAULT_FALLBACK_MS = 600;
// how long we trust a stale face box after the face detector loses the face
const FACE_STALE_MS = 1200;

// how far the head has to tilt laterally (roll, in degrees: ear to shoulder)
const MUERTE_ROLL_DEG = 20.0;

// hand-covering-face: how close the hand needs to be to where the mouth
// last was. Wider when the face detector has fully lost the face (strong
// evidence of a real occlusion); tighter when the face is still partially
// tracked (weaker evidence, avoid false positives from a hand just passing
// near the face).
const HAND_COVER_FACE_DIST_FACE_LOST = 1.3;
const HAND_COVER_FACE_DIST_FACE_SEEN = 0.7;

const video = document.getElementById("video");
const memeImg = document.getElementById("memeImg");
const debugHud = document.getElementById("debugHud");

let handLandmarker, faceLandmarker;
let lastVideoTime = -1;
let currentGesture = "emperador";
let candidateGesture = "emperador";
let candidateStreak = 0;
let lastNonDefaultAt = performance.now();
let lastFace = null; // { mouthCenter, faceWidth, rightCheek, leftCheek, forehead, rollDeg, yawDeg, t }
let lastFaceSeenThisFrame = false;
let lastRollDebug = 0;
let lastYawDebug = 0;

async function init() {
  const fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  handLandmarker = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
  });

  faceLandmarker = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFacialTransformationMatrixes: true,
  });

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480 },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();

  requestAnimationFrame(loop);
}

// ---- 3D-aware geometry helpers -----------------------------------------
// Using z (depth) as well as x/y makes these tests far more robust to hand
// rotation, foreshortening, and motion blur than a plain 2D/wrist-distance
// check would be.
function vec(a, b) {
  return { x: b.x - a.x, y: b.y - a.y, z: (b.z || 0) - (a.z || 0) };
}
function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, (a.z || 0) - (b.z || 0));
}
function angleDeg(v1, v2) {
  const dot = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
  const m1 = Math.hypot(v1.x, v1.y, v1.z);
  const m2 = Math.hypot(v2.x, v2.y, v2.z);
  if (m1 < 1e-9 || m2 < 1e-9) return 180;
  return (Math.acos(Math.min(1, Math.max(-1, dot / (m1 * m2)))) * 180) / Math.PI;
}

// a finger is "extended" if its two segments (mcp->pip, pip->tip) point in
// roughly the same direction; "curled" if it folds back sharply.
function fingerExtended(lm, mcp, pip, tip) {
  const angle = angleDeg(vec(lm[mcp], lm[pip]), vec(lm[pip], lm[tip]));
  return angle < 45;
}

// extract the head's lateral tilt angle (roll, degrees: ear to shoulder) from
// MediaPipe's facial transformation matrix.
function rollFromTransformMatrix(matrixData) {
  // matrixData is a 16-element row-major 4x4 array; r(row, col) = data[row*4+col]
  const r00 = matrixData[0];
  const r10 = matrixData[4];
  return (Math.atan2(r10, r00) * 180) / Math.PI;
}

// extract the head's left/right turn angle (yaw, degrees) from MediaPipe's
// facial transformation matrix.
function yawFromTransformMatrix(matrixData) {
  // matrixData is a 16-element row-major 4x4 array; r(row, col) = data[row*4+col]
  const r00 = matrixData[0];
  const r10 = matrixData[4];
  const r20 = matrixData[8];
  const sy = Math.hypot(r00, r10);
  if (sy < 1e-6) return 0;
  return (Math.atan2(-r20, sy) * 180) / Math.PI;
}

function classifyHand(lm) {
  const handScale = dist(lm[0], lm[9]) || 1e-6; // wrist -> middle mcp

  const indexUp = fingerExtended(lm, 5, 6, 8);
  const middleUp = fingerExtended(lm, 9, 10, 12);
  const ringUp = fingerExtended(lm, 13, 14, 16);
  const pinkyUp = fingerExtended(lm, 17, 18, 20);

  // thumb + pinky spread apart from each other = shaka/rock-on shape.
  // tucked thumb sits close to the pinky-side of the palm; an abducted
  // thumb sticks straight out and this distance grows a lot.
  const thumbPinkySpread = dist(lm[4], lm[17]) / handScale;
  const thumbOut = thumbPinkySpread > 1.05;

  const curledCount = [indexUp, middleUp, ringUp, pinkyUp].filter((v) => !v).length;

  return {
    indexUp,
    middleUp,
    ringUp,
    pinkyUp,
    thumbOut,
    curledCount,
    handScale,
    wrist: lm[0],
    thumbTip: lm[4],
    indexTip: lm[8],
    middleTip: lm[12],
    ringTip: lm[16],
    pinkyTip: lm[20],
    palmCenter: lm[9],
  };
}

function updateFace(faceResult) {
  const now = performance.now();
  const sawFace = !!(faceResult.faceLandmarks && faceResult.faceLandmarks.length > 0);

  if (sawFace) {
    const f = faceResult.faceLandmarks[0];
    const upperLip = f[13];
    const lowerLip = f[14];
    const rightCheek = f[234];
    const leftCheek = f[454];
    const forehead = f[10];
    const mouthCenter = {
      x: (upperLip.x + lowerLip.x) / 2,
      y: (upperLip.y + lowerLip.y) / 2,
      z: ((upperLip.z || 0) + (lowerLip.z || 0)) / 2,
    };
    const faceWidth = dist(rightCheek, leftCheek);
    const mouthOpen = dist(upperLip, lowerLip) / faceWidth;

    let rollDeg = 0;
    let yawDeg = 0;
    if (faceResult.facialTransformationMatrixes && faceResult.facialTransformationMatrixes.length > 0) {
      const mat = faceResult.facialTransformationMatrixes[0].data;
      rollDeg = rollFromTransformMatrix(mat);
      yawDeg = yawFromTransformMatrix(mat);
    } else {
      const dx = leftCheek.x - rightCheek.x;
      const dy = leftCheek.y - rightCheek.y;
      rollDeg = (Math.atan2(dy, dx) * 180) / Math.PI;
    }

    lastFace = { mouthCenter, faceWidth, rightCheek, leftCheek, forehead, mouthOpen, rollDeg, yawDeg, t: now };
    lastRollDebug = rollDeg;
    lastYawDebug = yawDeg;
  }
  lastFaceSeenThisFrame = sawFace;
}

function isPointing(h) {
  return h.indexUp && !h.middleUp && !h.ringUp && !h.pinkyUp;
}

function decideGesture(handResult) {
  const now = performance.now();
  const faceIsFresh = !!lastFace && now - lastFace.t < FACE_STALE_MS;

  // 1. Muerte — Cabeza inclinada lateralmente ≥20°, con una oreja acercándose al hombro
  if (faceIsFresh && Math.abs(lastFace.rollDeg) >= MUERTE_ROLL_DEG) {
    return "muerte";
  }

  if (!handResult.landmarks || handResult.landmarks.length === 0) {
    return "emperador";
  }

  const hands = handResult.landmarks.map(classifyHand);
  const mouthCenter = faceIsFresh ? lastFace.mouthCenter : { x: 0.5, y: 0.5, z: 0 };
  const faceWidth = faceIsFresh ? lastFace.faceWidth : 0.2;
  const headTopY = faceIsFresh ? (mouthCenter.y - faceWidth * 0.9) : 0.35;

  if (hands.length === 2) {
    const avgScale = (hands[0].handScale + hands[1].handScale) / 2;

    // 2. Sol — Ambos brazos levantados, con las manos sobre la cabeza y las puntas de los dedos enfrentadas formando un arco
    const solIndexGap = dist(hands[0].indexTip, hands[1].indexTip) / avgScale;
    const solThumbGap = dist(hands[0].thumbTip, hands[1].thumbTip) / avgScale;
    const solMiddleGap = dist(hands[0].middleTip, hands[1].middleTip) / avgScale;
    const wristGap = dist(hands[0].wrist, hands[1].wrist);
    const palmGap = dist(hands[0].palmCenter, hands[1].palmCenter);
    const indexDist = dist(hands[0].indexTip, hands[1].indexTip);

    const bothHandsAboveHead =
      hands[0].palmCenter.y < headTopY + 0.15 &&
      hands[1].palmCenter.y < headTopY + 0.15;

    const isSolPose =
      bothHandsAboveHead &&
      hands[0].curledCount <= 2 &&
      hands[1].curledCount <= 2 &&
      (solIndexGap < 2.8 || solMiddleGap < 2.8 || solThumbGap < 3.2) &&
      wristGap > indexDist * 1.05;

    // 4. Amantes — Ambas manos frente al pecho, índices y pulgares unidos formando un corazón
    const heartIndexGap = dist(hands[0].indexTip, hands[1].indexTip) / avgScale;
    const heartThumbGap = dist(hands[0].thumbTip, hands[1].thumbTip) / avgScale;
    const isAmantesPose =
      heartIndexGap < 1.6 &&
      heartThumbGap < 1.8 &&
      wristGap > indexDist * 1.15 &&
      hands[0].palmCenter.y > mouthCenter.y - 0.1 &&
      hands[1].palmCenter.y > mouthCenter.y - 0.1;

    // 5. Diablo — Ambas manos junto a la parte superior de la cabeza, con ambos índices extendidos hacia arriba
    const bothHandsAtHead =
      hands[0].palmCenter.y < mouthCenter.y + 0.1 &&
      hands[1].palmCenter.y < mouthCenter.y + 0.1;
    const bothIndexUp = hands[0].indexUp && hands[1].indexUp;
    const indexPointingUp =
      hands[0].indexTip.y < hands[0].palmCenter.y &&
      hands[1].indexTip.y < hands[1].palmCenter.y;
    const otherFingersCurled = hands[0].curledCount >= 2 && hands[1].curledCount >= 2;
    const handsSpreadHead = palmGap / faceWidth > 0.6;
    const isDiabloPose =
      bothHandsAtHead &&
      bothIndexUp &&
      indexPointingUp &&
      otherFingersCurled &&
      handsSpreadHead;

    // 6. El Loco — Ambos índices extendidos apuntando hacia las sienes, uno a cada lado de la cabeza
    const rightCheek = faceIsFresh ? lastFace.rightCheek : { x: 0.4, y: 0.4, z: 0 };
    const leftCheek = faceIsFresh ? lastFace.leftCheek : { x: 0.6, y: 0.4, z: 0 };
    const d1r = dist(hands[0].indexTip, rightCheek) / faceWidth;
    const d1l = dist(hands[0].indexTip, leftCheek) / faceWidth;
    const d2r = dist(hands[1].indexTip, rightCheek) / faceWidth;
    const d2l = dist(hands[1].indexTip, leftCheek) / faceWidth;
    const nearTemples = (d1r < 1.2 && d2l < 1.2) || (d1l < 1.2 && d2r < 1.2);
    const isLocoPose =
      bothIndexUp &&
      otherFingersCurled &&
      nearTemples &&
      !indexPointingUp;

    // Evaluar según jerarquía: Sol -> Mago (2 manos) -> Amantes -> Diablo -> El Loco
    if (isSolPose && !isAmantesPose) {
      return "sol";
    }

    // 3. Mago con 2 manos (un brazo arriba por encima de la cabeza y el otro abajo)
    const handsAboveHead = hands.filter(
      (h) => h.palmCenter.y < headTopY && h.wrist.y > h.palmCenter.y
    );
    if (handsAboveHead.length === 1) {
      const otherHand = hands.find((h) => h !== handsAboveHead[0]);
      if (otherHand && otherHand.palmCenter.y >= headTopY) {
        return "mago";
      }
    }

    if (isAmantesPose) {
      return "amantes";
    }

    if (isDiabloPose) {
      return "diablo";
    }

    if (isLocoPose) {
      return "elLoco";
    }
  }

  // Gestos con 1 mano visible
  if (hands.length === 1) {
    const h = hands[0];
    // 3. Mago — Un brazo extendido verticalmente hacia arriba, con la mano por encima de la cabeza
    if (h.palmCenter.y < headTopY && h.wrist.y > h.palmCenter.y) {
      return "mago";
    }
  }

  // 7. Emperador — Cabeza frontal, manos abajo y brazos relajados a ambos lados del cuerpo (Default)
  return "emperador";
}

function pickImage(gesture) {
  const images = GESTURE_MEMES[gesture];
  return images[Math.floor(Math.random() * images.length)];
}

function applyGesture(gesture) {
  if (gesture === currentGesture) return;
  currentGesture = gesture;
  memeImg.src = pickImage(gesture);
}

function loop() {
  const now = performance.now();
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const ts = performance.now();

    const handResult = handLandmarker.detectForVideo(video, ts);
    const faceResult = faceLandmarker.detectForVideo(video, ts);
    updateFace(faceResult);

    const gesture = decideGesture(handResult);

    // debounce: require a gesture to be seen for several consecutive
    // frames before we commit to it, to avoid flicker between frames
    if (gesture === candidateGesture) {
      candidateStreak++;
    } else {
      candidateGesture = gesture;
      candidateStreak = 1;
    }

    if (candidateStreak >= STABLE_FRAMES_REQUIRED) {
      applyGesture(gesture);
    }

    if (gesture !== "emperador") lastNonDefaultAt = now;
    if (now - lastNonDefaultAt > DEFAULT_FALLBACK_MS && currentGesture !== "emperador") {
      applyGesture("emperador");
    }

    updateDebugHud();
  }
  requestAnimationFrame(loop);
}

function updateDebugHud() {
  if (!debugHud) return;
  debugHud.textContent =
    `Carta: ${currentGesture}\n` +
    `Inclinacion (Roll): ${lastRollDebug >= 0 ? "+" : ""}${lastRollDebug.toFixed(1)} deg  (Muerte thr +/-${MUERTE_ROLL_DEG.toFixed(1)})`;
}

init().catch((err) => console.error(err));

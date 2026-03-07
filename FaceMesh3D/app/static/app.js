/**
 * FaceMesh3D — frontend
 * Camera capture  →  FastAPI backend  →  DECA 3D mesh  →  Three.js viewer
 */

import * as THREE from 'three';
import { OBJLoader }      from 'three/addons/loaders/OBJLoader.js';
import { MTLLoader }      from 'three/addons/loaders/MTLLoader.js';
import { OrbitControls }  from 'three/addons/controls/OrbitControls.js';

// ── DOM refs ──────────────────────────────────────────────────────────────────
const video            = document.getElementById('video');
const snapshotCanvas   = document.getElementById('snapshot-canvas');
const placeholder      = document.getElementById('camera-placeholder');
const countdownOverlay = document.getElementById('countdown-overlay');
const countdownNum     = document.getElementById('countdown-num');

const btnCameraToggle  = document.getElementById('btn-camera-toggle');
const btnFlip          = document.getElementById('btn-flip');
const btnSelfie        = document.getElementById('btn-selfie');
const btnRetake        = document.getElementById('btn-retake');
const btnReconstruct   = document.getElementById('btn-reconstruct');
const btnNew           = document.getElementById('btn-new');
const btnAutoRotate    = document.getElementById('btn-auto-rotate');
const btnDownloadObj   = document.getElementById('btn-download-obj');

const previewImg       = document.getElementById('preview-img');
const originalImg      = document.getElementById('original-img');
const viewImage        = document.getElementById('view-image');
const viewTabs         = document.getElementById('view-tabs');
const processingLabel  = document.getElementById('processing-label');

const toast            = document.getElementById('toast');

const SEC_CAMERA     = document.getElementById('section-camera');
const SEC_PREVIEW    = document.getElementById('section-preview');
const SEC_PROCESSING = document.getElementById('section-processing');
const SEC_RESULTS    = document.getElementById('section-results');

// ── State ─────────────────────────────────────────────────────────────────────
let stream       = null;
let facingMode   = 'user';   // 'user' = front, 'environment' = back
let cameraOn     = false;
let capturedBlob = null;
let threeViewer  = null;     // Three.js viewer instance
let autoRotate   = true;

// ── Section helper ────────────────────────────────────────────────────────────
function showSection(el) {
  [SEC_CAMERA, SEC_PREVIEW, SEC_PROCESSING, SEC_RESULTS].forEach(s => {
    s.classList.remove('active');
    s.hidden = true;
  });
  el.classList.add('active');
  el.hidden = false;
}

// ── Toast helper ──────────────────────────────────────────────────────────────
let toastTimer;
function showToast(msg) {
  toast.textContent = msg;
  toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toast.hidden = true; }, 5000);
}

// ── Camera ────────────────────────────────────────────────────────────────────
async function startCamera() {
  try {
    if (stream) stopCamera();
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode,
        width:  { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    cameraOn = true;
    placeholder.hidden = true;
    video.hidden = false;
    btnCameraToggle.innerHTML = '🔴 Stop Camera';
    btnFlip.disabled   = false;
    btnSelfie.disabled = false;
  } catch (err) {
    showToast(`Camera error: ${err.message}`);
  }
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  cameraOn = false;
  video.srcObject = null;
  video.hidden = true;
  placeholder.hidden = false;
  btnCameraToggle.innerHTML = '📷 Start Camera';
  btnFlip.disabled   = true;
  btnSelfie.disabled = true;
}

async function flipCamera() {
  facingMode = facingMode === 'user' ? 'environment' : 'user';
  // mirror only front camera
  video.style.transform = facingMode === 'user' ? 'scaleX(-1)' : 'scaleX(1)';
  await startCamera();
}

// ── Countdown + capture ───────────────────────────────────────────────────────
async function countdown(from = 3) {
  countdownOverlay.hidden = false;
  for (let i = from; i >= 1; i--) {
    countdownNum.textContent = i;
    await sleep(900);
  }
  countdownOverlay.hidden = true;
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function captureFrame() {
  const w = video.videoWidth  || 640;
  const h = video.videoHeight || 480;
  snapshotCanvas.width  = w;
  snapshotCanvas.height = h;
  const ctx = snapshotCanvas.getContext('2d');
  // Un-mirror so the server sees the real face
  if (facingMode === 'user') {
    ctx.translate(w, 0);
    ctx.scale(-1, 1);
  }
  ctx.drawImage(video, 0, 0, w, h);
  return new Promise(resolve => snapshotCanvas.toBlob(resolve, 'image/jpeg', 0.92));
}

async function takeSelfie() {
  btnSelfie.disabled = true;
  await countdown(3);
  capturedBlob = await captureFrame();
  const url = URL.createObjectURL(capturedBlob);
  previewImg.src = url;
  stopCamera();
  showSection(SEC_PREVIEW);
}

// ── Reconstruction request ────────────────────────────────────────────────────
const PROCESSING_LABELS = [
  'Detecting face…',
  'Fitting FLAME model…',
  'Predicting expression…',
  'Recovering texture…',
  'Building 3D mesh…',
  'Almost there…',
];

async function startReconstruction() {
  showSection(SEC_PROCESSING);

  // Cycle informational labels while waiting
  let labelIdx = 0;
  processingLabel.textContent = PROCESSING_LABELS[0];
  const labelTimer = setInterval(() => {
    labelIdx = (labelIdx + 1) % PROCESSING_LABELS.length;
    processingLabel.textContent = PROCESSING_LABELS[labelIdx];
  }, 4000);

  const formData = new FormData();
  formData.append('file', capturedBlob, 'selfie.jpg');

  let data;
  try {
    const res = await fetch('/api/reconstruct', { method: 'POST', body: formData });
    data = await res.json();
    if (!res.ok) throw new Error(data.detail || `Server error ${res.status}`);
  } catch (err) {
    clearInterval(labelTimer);
    showSection(SEC_CAMERA);
    showToast(`❌ ${err.message}`);
    return;
  }
  clearInterval(labelTimer);

  // Populate results UI
  populateResults(data);
  showSection(SEC_RESULTS);
}

// ── Results ───────────────────────────────────────────────────────────────────
function populateResults(data) {
  // Original photo
  originalImg.src = data.input_url;

  // Rendered-view tabs
  viewTabs.innerHTML = '';
  if (data.views && data.views.length > 0) {
    data.views.forEach((view, i) => {
      const btn = document.createElement('button');
      btn.className = 'tab-btn' + (i === 0 ? ' active' : '');
      btn.textContent = view.name;
      btn.addEventListener('click', () => {
        viewTabs.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        viewImage.src = view.data;
      });
      viewTabs.appendChild(btn);
    });
    viewImage.src = data.views[0].data;
  }

  // Download OBJ link
  btnDownloadObj.href = data.obj_url;

  // Three.js viewer
  initThreeViewer(data.obj_url, data.mtl_url);
}

// ── Three.js 3D viewer ────────────────────────────────────────────────────────
function initThreeViewer(objUrl, mtlUrl) {
  const container = document.getElementById('three-container');
  container.innerHTML = '';

  const W = container.clientWidth  || 400;
  const H = container.clientHeight || 400;

  // Scene
  const scene    = new THREE.Scene();
  scene.background = new THREE.Color(0x0d1020);

  // Camera
  const camera = new THREE.PerspectiveCamera(45, W / H, 0.01, 100);
  camera.position.set(0, 0, 2.5);

  // Renderer
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(W, H);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  container.appendChild(renderer.domElement);

  // Lights
  scene.add(new THREE.AmbientLight(0xffffff, 0.55));

  const keyLight = new THREE.DirectionalLight(0xffeedd, 1.0);
  keyLight.position.set(1.5, 2, 2);
  scene.add(keyLight);

  const fillLight = new THREE.DirectionalLight(0xddeeff, 0.5);
  fillLight.position.set(-2, 0, 1);
  scene.add(fillLight);

  const rimLight = new THREE.DirectionalLight(0xffffff, 0.3);
  rimLight.position.set(0, -2, -2);
  scene.add(rimLight);

  // Orbit controls
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping    = true;
  controls.dampingFactor    = 0.08;
  controls.minDistance      = 0.8;
  controls.maxDistance      = 8;
  controls.autoRotate       = autoRotate;
  controls.autoRotateSpeed  = 1.8;

  // Material fallback (used if MTL fails or has no texture)
  const defaultMat = new THREE.MeshPhongMaterial({
    color:     0xddccbb,
    specular:  0x222222,
    shininess: 30,
    side:      THREE.DoubleSide,
  });

  // Load mesh — try MTL + OBJ first, fall back to OBJ alone
  const loadObj = (materials) => {
    const objLoader = new OBJLoader();
    if (materials) objLoader.setMaterials(materials);

    objLoader.load(
      objUrl,
      (object) => {
        // If no material loaded, apply default
        object.traverse(child => {
          if (child.isMesh && (!materials || !child.material?.map)) {
            child.material = defaultMat;
          }
          if (child.isMesh) child.castShadow = true;
        });

        // Centre + scale the mesh
        const box    = new THREE.Box3().setFromObject(object);
        const centre = box.getCenter(new THREE.Vector3());
        const size   = box.getSize(new THREE.Vector3()).length();
        object.position.sub(centre);
        object.scale.setScalar(2 / size);

        // DECA mesh is upside-down — flip it
        object.rotation.x = Math.PI;

        scene.add(object);
      },
      undefined,
      (err) => console.warn('OBJ load error:', err),
    );
  };

  // Attempt to load MTL for texture
  const mtlLoader = new MTLLoader();
  const basePath  = mtlUrl.substring(0, mtlUrl.lastIndexOf('/') + 1);
  mtlLoader.setPath(basePath);
  mtlLoader.load(
    mtlUrl.split('/').pop(),
    (materials) => { materials.preload(); loadObj(materials); },
    undefined,
    () => loadObj(null),   // MTL failed — load without texture
  );

  // Store references for auto-rotate toggle
  threeViewer = { controls, renderer, scene, camera, animId: null };

  // Resize observer
  const ro = new ResizeObserver(() => {
    const w = container.clientWidth;
    const h = container.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  });
  ro.observe(container);

  // Render loop
  function animate() {
    threeViewer.animId = requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();
}

// ── Auto-rotate toggle ────────────────────────────────────────────────────────
btnAutoRotate.addEventListener('click', () => {
  autoRotate = !autoRotate;
  if (threeViewer) threeViewer.controls.autoRotate = autoRotate;
  btnAutoRotate.classList.toggle('active', autoRotate);
  btnAutoRotate.textContent = autoRotate ? '⏸ Auto-rotate' : '▶ Auto-rotate';
});

// ── Button wiring ─────────────────────────────────────────────────────────────
btnCameraToggle.addEventListener('click', () => {
  if (cameraOn) stopCamera();
  else startCamera();
});

btnFlip.addEventListener('click', flipCamera);

btnSelfie.addEventListener('click', () => {
  if (!cameraOn) return;
  takeSelfie();
});

btnRetake.addEventListener('click', () => {
  capturedBlob = null;
  showSection(SEC_CAMERA);
  startCamera();
});

btnReconstruct.addEventListener('click', startReconstruction);

btnNew.addEventListener('click', () => {
  // Destroy Three.js viewer
  if (threeViewer) {
    cancelAnimationFrame(threeViewer.animId);
    threeViewer.renderer.dispose();
    threeViewer = null;
  }
  capturedBlob = null;
  showSection(SEC_CAMERA);
  startCamera();
});

// Auto-start camera on page load
startCamera();

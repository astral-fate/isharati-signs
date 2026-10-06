// A VRM avatar driven by Isharati poses: [T,50,3] = 8 upper-body joints + 21 left-hand + 21 right-hand points,
// neck-centred, image axes (x right, y down, z away from the camera).
//
// Each bone is turned so that its rest direction (measured on the avatar in its T-pose, where normalized VRM bones
// have identity rotations) points along the matching keypoint segment: qLocal = fromUnitVectors(rest, parent⁻¹·d).
// The hands take a full orientation from two vectors (wrist→middle knuckle and index→little knuckle), so palm
// facing is kept; fingers then swing inside the hand. Body depth from MediaPipe is noisy, so it is damped and arms
// are kept from pointing behind the body. Rotations are smoothed between frames.
//
// Fixes applied (v2):
//  • Quaternion hemisphere continuity: before slerp, negate q if dot(prev,q) < 0, preventing the 300° "wrong way"
//    path that caused wrists to flip to a physically-impossible orientation.
//  • Wrist orient fallback: if the two-vector frame produces a rotation > ORIENT_FLIP_RAD from the previous frame,
//    discard it and keep the smoothed previous value (short-circuits single-frame spikes).
//  • Finger-segment vectors that are too short (< MIN_SEG) or point wildly backward relative to the hand frame are
//    clamped so they cannot produce crossed or hyper-extended fingers.
//  • SMOOTH raised to 0.72 (was 0.55) so the filter is heavier — fast glitches are suppressed more.
//  • Hand collapse threshold raised so a partially-visible hand still updates rather than freezing.
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { VRMLoaderPlugin, VRMUtils } from "@pixiv/three-vrm";

const NOSE = 0, NECK = 1, R_SH = 2, R_EL = 3, R_WR = 4, L_SH = 5, L_EL = 6, L_WR = 7, LH = 8, RH = 29;
const FINGERS = { Thumb: [1, 2, 3, 4], Index: [5, 6, 7, 8], Middle: [9, 10, 11, 12], Ring: [13, 14, 15, 16],
                  Little: [17, 18, 19, 20] };
const SEGMENTS = { Thumb: ["Metacarpal", "Proximal", "Distal"], Index: ["Proximal", "Intermediate", "Distal"],
                   Middle: ["Proximal", "Intermediate", "Distal"], Ring: ["Proximal", "Intermediate", "Distal"],
                   Little: ["Proximal", "Intermediate", "Distal"] };
const BODY_DEPTH   = 0.5;    // MediaPipe body z is rough: halve it
const SMOOTH       = 0.72;   // share of the new rotation taken each frame  (was 0.55 — heavier smoothing)
const MIN_SEG      = 1e-3;   // minimum finger-segment length to trust; below this keep previous
const ORIENT_FLIP  = 2.2;    // radians — if the new wrist orientation jumps more than this, discard the frame
const MAX_REJECT   = 3;      // ...but never more than this many frames in a row
const FACE_SMOOTH  = 0.6;
const WRIST_MAX    = 65 * Math.PI / 180;  // a wrist bends at most ~65-80° from the forearm line    // share of the new blendshape score taken each frame

const DEG = Math.PI / 180;
const MCP_FLEX = [-20 * DEG, 95 * DEG], MCP_SPREAD = 25 * DEG;     // knuckle: fold range, sideways spread
const PIP_MAX = 110 * DEG, DIP_MAX = 85 * DEG, HINGE_BACK = 10 * DEG;
const clamp = THREE.MathUtils.clamp;

// The first finger segment (knuckle -> middle joint) limited in the palm's frame: fwd along the fingers, n out of the
// palm, across the palm. Fold (toward n) and spread (along across) are clamped, and the direction rebuilt.
function limitKnuckle(seg, fwd, n, across) {
  const s = seg.clone().normalize();
  const flex = clamp(Math.atan2(s.dot(n), s.dot(fwd)), MCP_FLEX[0], MCP_FLEX[1]);
  const spread = clamp(Math.asin(clamp(s.dot(across), -1, 1)), -MCP_SPREAD, MCP_SPREAD);
  return fwd.clone().multiplyScalar(Math.cos(flex)).addScaledVector(n, Math.sin(flex)).multiplyScalar(Math.cos(spread))
    .addScaledVector(across, Math.sin(spread)).normalize();
}

// A hinge joint after segment u: the next segment folds only in the finger's own flexion plane (no sideways bend), by
// an angle clamped to [-HINGE_BACK, max]. The hinge axis is the finger's side-to-side axis: the finger's direction in
// the palm plane crossed with n. (Folding "toward n" instead breaks down once the knuckle is bent 90°, when u points
// along n: every fist, such as alef's, then lost the curl of its middle and end joints.)
function hinge(u, seg, n, max, across) {
  const inPalm = u.clone().addScaledVector(n, -u.dot(n));
  // past a 90° knuckle u leans back over the palm and inPalm would point toward the wrist, turning the hinge axis
  // (and the fold) around: keep it pointing along the fingers (forward = n x across)
  if (inPalm.dot(new THREE.Vector3().crossVectors(n, across)) < 0) inPalm.negate();
  const axis = inPalm.lengthSq() > 1e-4 ? new THREE.Vector3().crossVectors(inPalm, n).normalize() : across.clone();
  const toFold = new THREE.Vector3().crossVectors(axis, u).normalize();  // = n when u lies along the fingers
  const s = seg.clone().normalize();
  const fold = clamp(Math.atan2(s.dot(toFold), s.dot(u)), -HINGE_BACK, max);
  return u.clone().multiplyScalar(Math.cos(fold)).addScaledVector(toFold, Math.sin(fold)).normalize();
}

// Finger curl as a whole, for when the joints themselves cannot be trusted. In a closed fist MediaPipe often puts the
// middle joints behind the knuckles (KArSL's alef: knuckles "folded" 137-141°, middle joints ±160-170°), which no
// per-joint reading can recover. The fingertip's distance from its knuckle, over the finger's length, survives that
// noise; a curl angle T is spread over the joints as a hand closes (knuckle 38%, middle 42%, end 20%).
const CURL_SHARE = [0.38, 0.42, 0.20], CURL_LIMIT = [MCP_FLEX[1], PIP_MAX, DIP_MAX], CURL_LEN = [0.45, 0.32, 0.23];
const CURL_TABLE = (() => {                                          // [tip-to-knuckle / length, T] for T = 0..260°
  const out = [];
  for (let T = 0; T <= 260; T += 2) {
    let x = 0, y = 0, a = 0;
    for (let k = 0; k < 3; k++) {
      a += Math.min(CURL_SHARE[k] * T * DEG, CURL_LIMIT[k]);
      x += CURL_LEN[k] * Math.cos(a);
      y += CURL_LEN[k] * Math.sin(a);
    }
    out.push([Math.hypot(x, y), T * DEG]);
  }
  return out;
})();

// How open a finger is: its tip's distance from its knuckle over the finger's expected length, taken from the palm
// (wrist -> middle knuckle) rather than from the measured segments, whose zig-zag noise would make an open finger look
// curled. 1 = straight, about 0.35 = a fist.
// A finger tilted toward the camera looks short against the palm although its joints are straight (KArSL «الصلاة»:
// palm-based 0.77, joints 0.95), which rebuilt it as a claw. So the finger's own straightness (knuckle-to-tip over the
// sum of its segments, which foreshortening does not change) also counts: a fist is low on both measures.
const FINGER_LEN = { Index: 0.88, Middle: 0.95, Ring: 0.88, Little: 0.72 };
function openness(segs, palmLen, finger) {
  const chord = segs.reduce((s, d) => s.add(d), new THREE.Vector3()).length();
  const arc = segs.reduce((s, d) => s + d.length(), 0);
  return clamp(Math.max(chord / (FINGER_LEN[finger] * palmLen), arc > 1e-6 ? chord / arc : 0), 0, 1);
}

// the openness a set of finger directions gives, with the model's segment proportions
function modelOpenness(dirs) {
  return dirs.reduce((s, d, k) => s.addScaledVector(d, CURL_LEN[k]), new THREE.Vector3()).length();
}

function curlFinger(segs, fwd, n, across, ratio) {
  const T = CURL_TABLE.reduce((best, r) => (Math.abs(r[0] - ratio) < Math.abs(best[0] - ratio) ? r : best))[1];
  // spread is read from the first segment only while the finger is fairly open; a curled finger points along n
  const s0 = segs[0].clone().normalize();
  const spread = ratio > 0.7 ? clamp(Math.asin(clamp(s0.dot(across), -1, 1)), -MCP_SPREAD, MCP_SPREAD) : 0;
  const base = fwd.clone().multiplyScalar(Math.cos(spread)).addScaledVector(across, Math.sin(spread)).normalize();
  const axis = new THREE.Vector3().crossVectors(base, n).normalize();
  const dirs = [];
  let a = 0;
  for (let k = 0; k < 3; k++) {
    a += Math.min(CURL_SHARE[k] * T, CURL_LIMIT[k]);
    dirs.push(base.clone().applyAxisAngle(axis, a));                 // axis x base = n: folds toward the palm
  }
  return dirs;
}

// Are the finger's joints, as measured, something a hand can do? Then they are kept as they are (finer handshapes);
// otherwise the finger is rebuilt from its overall curl.
function plausibleFinger(segs, fwd, n, across) {
  const s0 = segs[0].clone().normalize();
  const mcp = Math.atan2(s0.dot(n), s0.dot(fwd));
  if (mcp < MCP_FLEX[0] - 5 * DEG || mcp > MCP_FLEX[1] + 5 * DEG) return false;
  let u = s0;
  for (const [k, max] of [[1, PIP_MAX], [2, DIP_MAX]]) {
    const s = segs[k].clone().normalize();
    const inPalm = u.clone().addScaledVector(n, -u.dot(n));
    if (inPalm.dot(fwd) < 0) inPalm.negate();                     // as in hinge(): forward, past a 90° knuckle
    const axis = inPalm.lengthSq() > 1e-4 ? new THREE.Vector3().crossVectors(inPalm, n).normalize() : across;
    const toFold = new THREE.Vector3().crossVectors(axis, u).normalize();
    const fold = Math.atan2(s.dot(toFold), s.dot(u));
    if (Math.abs(s.dot(axis)) > Math.sin(30 * DEG) || fold < -HINGE_BACK - 5 * DEG || fold > max + 5 * DEG) return false;
    u = s;
  }
  return true;
}

const v   = (p) => new THREE.Vector3(p[0], -p[1], -p[2]);          // image axes → three.js axes
const sub = (a, b) => new THREE.Vector3().subVectors(a, b);

// Return a quaternion on the same hemisphere as ref (so slerp always takes the short arc).
// Note: Three.js Quaternion does not have a .negate() method, so we negate components directly.
function ensureContinuity(q, ref) {
  if (!ref) return q;
  return ref.dot(q) < 0 ? new THREE.Quaternion(-q.x, -q.y, -q.z, -q.w) : q;
}

// Build an orthonormal frame from two vectors (Gram–Schmidt).
// Returns a Matrix4 whose columns are [X, Y, Z] = [x̂, ŷ, ẑ].
function makeFrame(x, y) {
  const X = x.clone().normalize();
  const Z = new THREE.Vector3().crossVectors(X, y).normalize();
  // If x and y are nearly parallel, Z will be degenerate — catch it.
  if (Z.lengthSq() < 0.01) return null;
  return new THREE.Matrix4().makeBasis(X, new THREE.Vector3().crossVectors(Z, X), Z);
}

export class SignAvatar {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(26, 1, 0.1, 20);
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x445566, 1.6));
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(0.6, 1.6, 2.2);
    this.scene.add(key);
    // fill from the other side and a rim from behind: hands turned away from the key light, and dark skin or robes
    // against the dark panel, stay readable (the realistic avatars were near-black in those poses)
    const fill = new THREE.DirectionalLight(0xffffff, 0.9);
    fill.position.set(-1.2, 1.0, 2.0);
    this.scene.add(fill);
    const rim = new THREE.DirectionalLight(0xdfe9ff, 1.1);
    rim.position.set(0.3, 2.0, -2.5);
    this.scene.add(rim);
    this.lights = this.scene.children.filter((o) => o.isLight).map((l) => [l, l.intensity]);
    this.prev = new Map();
    this.rejects = new Map();
    this.frames = null;
  }

  async load(url) {
    const loader = new GLTFLoader();
    loader.register((parser) => new VRMLoaderPlugin(parser));
    const gltf = await loader.loadAsync(url);
    const vrm = gltf.userData.vrm;
    // The lights are set for the realistic (PBR) avatars, dark robes included; a toon-shaded (MToon) avatar under them
    // washes out to near-white skin and clothes, so it gets them dimmed
    let toon = false;
    gltf.scene.traverse((o) => { for (const m of [].concat(o.material || [])) if (m.isMToonMaterial) toon = true; });
    for (const [l, i] of this.lights) l.intensity = i * (toon ? 0.55 : 1);
    VRMUtils.removeUnnecessaryVertices(gltf.scene);
    VRMUtils.rotateVRM0(vrm);                      // VRM0 models face -Z; turn them to face the camera like VRM1
    this.vrm = vrm;
    this.scene.add(vrm.scene);
    vrm.scene.updateMatrixWorld(true);
    this.node = (name) => vrm.humanoid.getNormalizedBoneNode(name);
    const pos  = (name) => this.node(name).getWorldPosition(new THREE.Vector3());
    // rest directions, measured on the T-pose
    this.rest = {};
    for (const side of ["left", "right"]) {
      this.rest[`${side}UpperArm`] = sub(pos(`${side}LowerArm`), pos(`${side}UpperArm`)).normalize();
      this.rest[`${side}LowerArm`] = sub(pos(`${side}Hand`), pos(`${side}LowerArm`)).normalize();
      // rest hinge: the T-pose elbow folds the forearm forward (+Z, toward the camera)
      this.rest[`${side}Hinge`]    = new THREE.Vector3().crossVectors(this.rest[`${side}UpperArm`], new THREE.Vector3(0, 0, 1)).normalize();
      this.rest[`${side}HandA`]    = sub(pos(`${side}MiddleProximal`), pos(`${side}Hand`)).normalize();
      this.rest[`${side}HandB`]    = sub(pos(`${side}IndexProximal`), pos(`${side}LittleProximal`)).normalize();
      for (const [f, segs] of Object.entries(SEGMENTS)) {
        const names = segs.map((s) => `${side}${f}${s}`);
        for (let k = 0; k < names.length; k++) {
          const a = pos(names[k]);
          const b = k + 1 < names.length ? pos(names[k + 1]) : null;
          // the last bone has no child bone: continue the previous segment's direction
          this.rest[names[k]] = b ? sub(b, a).normalize() : sub(a, pos(names[k - 1])).normalize();
        }
      }
    }
    // rest pose of the bones outfits attach to, and the face size (outfits.js scales its pieces by it)
    this.restBone = {};
    for (const name of ["head", "neck", "upperChest", "chest", "hips", "leftShoulder", "rightShoulder",
                        "leftUpperArm", "leftLowerArm", "leftHand", "rightUpperArm", "rightLowerArm", "rightHand"]) {
      const n = this.node(name);
      if (n) this.restBone[name] = { pos: n.getWorldPosition(new THREE.Vector3()),
                                     quat: n.getWorldQuaternion(new THREE.Quaternion()) };
    }
    // the avatar's own body, for placing the hands (pose()): shoulder joints, neck point between them, face height
    const sh = { left: pos("leftUpperArm"), right: pos("rightUpperArm") };
    const neckPt = sh.left.clone().add(sh.right).multiplyScalar(0.5);
    const sw = sh.left.distanceTo(sh.right);
    const eyes = this.node("leftEye") && this.node("rightEye")
      ? pos("leftEye").add(pos("rightEye")).multiplyScalar(0.5) : pos("head").add(new THREE.Vector3(0, 0.25 * sw, 0));
    const nose = eyes.clone().add(new THREE.Vector3(0, -0.1 * sw, 0.08 * sw));  // nose tip: below and in front of the eyes
    this.body = { neck: neckPt, sw, noseH: nose.y - neckPt.y, faceZ: nose.z, chestZ: neckPt.z + 0.45 * sw, sh,
                  len: Object.fromEntries(["left", "right"].map((s) => [s, [
                    pos(`${s}UpperArm`).distanceTo(pos(`${s}LowerArm`)), pos(`${s}LowerArm`).distanceTo(pos(`${s}Hand`))]])),
                  palm: Object.fromEntries(["left", "right"].map((s) => [s, pos(`${s}Hand`).distanceTo(pos(`${s}MiddleProximal`))])) };
    const face = vrm.scene.getObjectByName("Face");
    this.faceBox = face ? new THREE.Box3().setFromObject(face) : null;
    // frame the upper body
    const chest = pos(this.node("upperChest") ? "upperChest" : "chest");
    const head  = pos("head");
    const h = head.y - chest.y;
    this.camera.position.set(0, chest.y + h * 0.3, h * 9.2);
    this.camera.lookAt(0, chest.y + h * 0.12, 0);   // from the top of the head to the waist
    vrm.expressionManager?.setValue("relaxed", 0.3);
    // ARKit face shapes (Rocketbox: AK_01_BrowDownLeft ... AK_52_TongueOut), keyed by MediaPipe's blendshape names
    // (browDownLeft ...): the signer's face then drives them one to one (setFace)
    this.faceMorphs = new Map();
    vrm.scene.traverse((o) => {
      for (const [key, idx] of Object.entries(o.morphTargetDictionary || {})) {
        const m = /AK_\d+_(\w+)$/.exec(key);
        if (!m) continue;
        const name = m[1][0].toLowerCase() + m[1].slice(1);
        if (!this.faceMorphs.has(name)) this.faceMorphs.set(name, []);
        this.faceMorphs.get(name).push([o, idx]);
      }
    });
    this.face = new Map();      // smoothed blendshape weights of the current frame
    this.resize();
    return this;
  }

  // The signer's facial expression for one frame: blendshape scores (MediaPipe face landmarker, ARKit names).
  // An avatar with ARKit shapes takes them as they are (applied after vrm.update, which would otherwise reset the
  // shapes its presets also use); any other VRM gets them folded into its presets: blinks, mouth shapes, emotions.
  setFace(row, names) {
    if (!row || !names) return;
    for (let k = 0; k < names.length; k++) {
      const v = row[k];
      if (v == null || Number.isNaN(v)) continue;               // no face in this frame: keep the last expression
      const p = this.face.get(names[k]) ?? 0;
      this.face.set(names[k], p + FACE_SMOOTH * (v - p));
    }
    if (this.faceMorphs.size || !this.vrm?.expressionManager) return;
    const f = (n) => this.face.get(n) ?? 0, avg = (a, b) => (f(a) + f(b)) / 2;
    const em = this.vrm.expressionManager;
    const set = (preset, v) => em.getExpression(preset) && em.setValue(preset, Math.min(1, Math.max(0, v)));
    set("relaxed", 0);
    set("blinkLeft", f("eyeBlinkLeft")); set("blinkRight", f("eyeBlinkRight"));
    set("aa", f("jawOpen")); set("ou", f("mouthPucker")); set("oh", f("mouthFunnel"));
    set("ee", avg("mouthStretchLeft", "mouthStretchRight"));
    set("happy", avg("mouthSmileLeft", "mouthSmileRight"));
    set("sad", avg("mouthFrownLeft", "mouthFrownRight"));
    set("angry", avg("browDownLeft", "browDownRight"));
    set("surprised", Math.max(f("browInnerUp"), avg("browOuterUpLeft", "browOuterUpRight")));
  }

  applyFaceMorphs() {
    for (const [name, targets] of this.faceMorphs) {
      const v = this.face.get(name);
      if (v === undefined) continue;
      for (const [mesh, idx] of targets) mesh.morphTargetInfluences[idx] = v;
    }
  }

  resize() {
    const w = this.canvas.clientWidth || 480, h = this.canvas.clientHeight || 480;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  setFrames(frames) { this.frames = frames; this.prev.clear(); this.rejects.clear(); this.springReset = true; }

  // swing the bone so its rest direction points along d (world space)
  aim(name, d) {
    const n = this.node(name);
    if (!n || d.lengthSq() < 1e-10) return;
    const parentQ = n.parent.getWorldQuaternion(new THREE.Quaternion());
    const local = d.clone().normalize().applyQuaternion(parentQ.invert());
    let q = new THREE.Quaternion().setFromUnitVectors(this.rest[name], local);
    q = ensureContinuity(q, this.prev.get(name));
    this.set(name, q);
  }

  // full orientation from two target vectors: a (primary) and b (roughly perpendicular)
  orient(name, restA, restB, a, b) {
    const n = this.node(name);
    if (!n || a.lengthSq() < 1e-10 || b.lengthSq() < 1e-10) return false;
    const mTarget = makeFrame(a, b);
    const mRest   = makeFrame(restA, restB);
    if (!mTarget || !mRest) return false;        // degenerate geometry — skip frame
    const world = new THREE.Quaternion().setFromRotationMatrix(
      mTarget.multiply(mRest.invert()));
    const parentQ = n.parent.getWorldQuaternion(new THREE.Quaternion());
    let q = parentQ.invert().multiply(world);
    // Hemisphere continuity — must be done BEFORE the flip-detection angle calculation
    const prev = this.prev.get(name);
    q = ensureContinuity(q, prev);
    // Discard the frame if the wrist jumped more than ORIENT_FLIP radians from the last accepted frame.
    // This catches single-frame spikes (hand tracking loss + recovery producing mirror flip).
    // At most MAX_REJECT frames in a row: a guard that always rejects locks a hand into whatever orientation it had
    // first (a noisy first frame left صوم's palm flipped, 147° off, for the whole sign).
    if (prev) {
      const angle = 2 * Math.acos(Math.min(1, Math.abs(prev.dot(q))));
      const rejected = this.rejects.get(name) || 0;
      if (angle > ORIENT_FLIP && rejected < MAX_REJECT) { this.rejects.set(name, rejected + 1); return false; }
    }
    this.rejects.set(name, 0);
    this.set(name, q);
    return true;
  }

  set(name, q) {
    const n = this.node(name);
    const p = this.prev.get(name);
    if (p) q = p.clone().slerp(q, SMOOTH);
    this.prev.set(name, q.clone());
    n.quaternion.copy(q);
    n.updateMatrixWorld(true);
  }

  // rotation taking one hand frame (forward along the fingers, palm normal) onto another
  toHand(n, fwd, nH, fH) {
    const m1 = new THREE.Matrix4().makeBasis(fwd, n, new THREE.Vector3().crossVectors(fwd, n).normalize());
    const m2 = new THREE.Matrix4().makeBasis(fH, nH, new THREE.Vector3().crossVectors(fH, nH).normalize());
    return new THREE.Quaternion().setFromRotationMatrix(m2.multiply(m1.invert()));
  }

  // A signer keypoint (three.js axes) placed on the avatar's body. The signer and the avatar differ in proportions
  // (cartoon heads are large, arms 0.7-1.0 shoulder widths long), so copying arm directions put hands that touch the
  // mouth at the neck. Instead the point keeps its place relative to the body landmarks: across in shoulder widths,
  // up in face heights above the neck (nose = 1) and down in shoulder widths, forward in shoulder widths.
  // s.neck.z must be on the same (damped) depth scale as p.z.
  place(p, s) {
    const B = this.body, dy = p.y - s.neck.y;
    return new THREE.Vector3(
      B.neck.x + (p.x - s.neck.x) / s.sw * B.sw,
      B.neck.y + (dy > 0 ? dy / s.noseH * B.noseH : dy / s.sw * B.sw),
      B.neck.z + (p.z - s.neck.z) / s.sw * B.sw);                  // depth: callers pass damped z
  }

  // Two-bone IK: turn upper and lower arm so the wrist reaches target; the elbow bends toward pole (signer's elbow).
  reach(side, target, pole, elbowAngle = Math.PI) {
    const S = this.body.sh[side], [L1, L2] = this.body.len[side];
    const D = sub(target, S);
    this.reachRatio = { ...(this.reachRatio || {}), [side]: D.length() / (L1 + L2) };  // > 1: target out of reach
    // A target beyond the avatar's arm was reached with the arm locked straight: the realistic avatars have arms of
    // ~1.3-1.4 shoulder widths (the cartoon 2.0), so in a sign made with a bent elbow (وضوء: ~100°) the Rocketbox arm
    // stood at 150-180°. The wrist goes no farther from the shoulder than the avatar's own arm reaches with the
    // signer's elbow angle (law of cosines on the avatar's bone lengths), so the elbow keeps the signer's bend; a
    // reachable target (a hand at the mouth or chest) is unchanged.
    const maxD = Math.sqrt(L1 * L1 + L2 * L2 - 2 * L1 * L2 * Math.cos(elbowAngle));
    if (D.length() > maxD) D.setLength(maxD);
    const d = THREE.MathUtils.clamp(D.length(), Math.abs(L1 - L2) + 1e-4, L1 + L2 - 1e-4);
    const dn = D.clone().normalize();
    let bend = sub(pole, S);
    bend.addScaledVector(dn, -bend.dot(dn));                      // pole direction, perpendicular to shoulder→wrist
    if (bend.lengthSq() < 1e-8) bend = new THREE.Vector3(0, -1, -0.3).addScaledVector(dn, -dn.y);
    bend.normalize();
    const a = (L1 * L1 - L2 * L2 + d * d) / (2 * d), h = Math.sqrt(Math.max(0, L1 * L1 - a * a));
    const E = S.clone().addScaledVector(dn, a).addScaledVector(bend, h);
    const W = S.clone().addScaledVector(dn, d);
    // An elbow is a hinge. Aiming each bone by the shortest swing left the upper arm's roll to chance, so the forearm
    // folded about whatever axis that gave: the cartoon's soft skinning hid it, the Rocketbox meshes (skinned for a
    // forward fold from the T-pose, palms down) showed a kinked, squashed elbow. So the upper arm takes its full
    // orientation, rolled until its rest hinge axis is the actual bend axis; the forearm then swings about that axis.
    const hinge = new THREE.Vector3().crossVectors(bend, dn);          // axis turning the upper arm onto the forearm
    if (!this.orient(`${side}UpperArm`, this.rest[`${side}UpperArm`], this.rest[`${side}Hinge`], sub(E, S), hinge))
      this.aim(`${side}UpperArm`, sub(E, S));
    this.aim(`${side}LowerArm`, sub(W, E));
  }

  pose(f) {
    const P = (i) => v(f[i]);
    const sL = P(L_SH), sR = P(R_SH), sNeck = sL.clone().add(sR).multiplyScalar(0.5);
    const sw = sL.distanceTo(sR) || 1;
    const signer = { neck: sNeck.clone().setZ(sNeck.z * BODY_DEPTH), sw, noseH: Math.max(0.3 * sw, P(NOSE).y - sNeck.y) };
    const B = this.body;
    for (const [side, sh, el, wr, base] of [["left", L_SH, L_EL, L_WR, LH], ["right", R_SH, R_EL, R_WR, RH]]) {
      // The hand is placed by its middle knuckle, the part that touches the face or body: the wrist target is that
      // point minus one avatar palm length along the signer's wrist→knuckle direction. (Hands are a different size
      // relative to the face on every source and avatar, so placing the wrist left the fingers short of the mouth.)
      // Hand landmarks come from a separate model whose depth has its own origin (in one PRAYER frame the body wrist
      // is at z -2.9 and the hand's wrist at +0.6), so the hand keeps its x, y and its depth relative to its own
      // wrist, and takes the body's (damped) wrist depth as a whole.
      const handWrist = P(base), handDir = sub(P(base + 9), handWrist);
      const hasHand = handDir.lengthSq() > 1e-8;
      const wrist = hasHand ? new THREE.Vector3(handWrist.x, handWrist.y, P(wr).z * BODY_DEPTH)
                            : P(wr).setZ(P(wr).z * BODY_DEPTH);
      wrist.z = THREE.MathUtils.clamp(wrist.z, sNeck.z * BODY_DEPTH, sNeck.z * BODY_DEPTH + 1.2 * sw);
      const knuckle = wrist.clone().add(handDir);
      const T = hasHand ? this.place(knuckle, signer).addScaledVector(handDir.normalize(), -B.palm[side])
                        : this.place(wrist, signer);
      const elbow = P(el); elbow.z *= BODY_DEPTH;
      const pole = this.place(elbow, signer);
      // keep the wrist in front of the body: in front of the face when it is up at the face, of the chest below
      const nearMid = Math.abs(T.x - B.neck.x) < 0.75 * B.sw;
      const atFace = T.y > B.neck.y + 0.35 * B.noseH;
      if (nearMid) T.z = Math.max(T.z, atFace ? B.faceZ + 0.15 * B.sw : B.chestZ);
      pole.z = Math.max(pole.z, B.neck.z - 0.2 * B.sw);             // the elbow never folds behind the back
      // the signer's elbow angle (180° = straight): the most the avatar's arm may open to reach T
      const up = sub(P(sh), P(el)), fore = sub(P(wr), P(el));
      const elbowAngle = up.lengthSq() > 1e-10 && fore.lengthSq() > 1e-10
        ? THREE.MathUtils.clamp(up.angleTo(fore), 0.35, Math.PI) : Math.PI;
      // at the face the touch matters more than the elbow angle (رمضان, صوم: fingers on the mouth or chin)
      this.reach(side, T, pole, nearMid && atFace ? Math.PI : elbowAngle);
      const H = (k) => v(f[base + k]);
      // Wrist→middle-knuckle (a) and index-knuckle→little-knuckle (b) define the palm plane.
      let a = sub(H(9), H(0)), b = sub(H(5), H(17));
      // Skip only if the hand is truly collapsed (all landmarks at origin / not detected).
      if (a.length() < 1e-4 && b.length() < 1e-4) continue;

      // Anti-compression normalization: in real human anatomy, palm width / length ratio stays between 0.45 and 1.25.
      // Mediapipe perspective distortion or noisy hand bounding box often squashes b or a into near-zero,
      // causing the avatar's palm to glitch or twist into an origami crease.
      const lenA = a.length(), lenB = b.length();
      if (lenA > 1e-3 && lenB > 1e-3) {
        const ratio = lenB / lenA;
        if (ratio < 0.45) {
          b.multiplyScalar(0.45 / ratio);
        } else if (ratio > 1.25) {
          b.multiplyScalar(1.25 / ratio);
        }
      }

      // A real wrist bends at most ~70° off the forearm. Noisy hand depth (or an avatar whose forearm sits a little
      // differently from the signer's) asked for 90°+, a hand snapped back at the wrist: tilt the whole hand frame
      // back toward the forearm line until it is within WRIST_MAX.
      const F = sub(this.node(`${side}Hand`).getWorldPosition(new THREE.Vector3()),
                    this.node(`${side}LowerArm`).getWorldPosition(new THREE.Vector3())).normalize();
      const bendAng = a.angleTo(F);
      if (bendAng > WRIST_MAX) {
        const axis = new THREE.Vector3().crossVectors(a, F);
        if (axis.lengthSq() > 1e-10) {
          const fix = new THREE.Quaternion().setFromAxisAngle(axis.normalize(), bendAng - WRIST_MAX);
          a.applyQuaternion(fix); b.applyQuaternion(fix);
        }
      }

      // orient() returns false on geometry failure or flip spike: the hand keeps its last orientation, and the fingers
      // (rotations relative to the hand) are still posed
      this.orient(`${side}Hand`, this.rest[`${side}HandA`], this.rest[`${side}HandB`], a, b);
      // Fingers as a hand can hold them. MediaPipe's finger depth is noisy, and 37-67% of hand-up frames in every
      // source had a middle or end joint bent over 30° sideways or 25° backwards (scripts/avatars/
      // finger_plausibility.py), which the avatar copied into twisted, broken-looking fingers. So each finger is
      // rebuilt joint by joint: the knuckle folds -20..95° and spreads ±25° in the palm's frame; the middle and end
      // joints are hinges that only fold toward the palm (0..110°, 0..85°, 10° back at most), never sideways.
      const n = (side === "left" ? new THREE.Vector3().crossVectors(a, b) : new THREE.Vector3().crossVectors(b, a)).normalize();
      const fwd = a.clone().normalize(), across = new THREE.Vector3().crossVectors(fwd, n).normalize();
      for (const [finger, idx] of Object.entries(FINGERS)) {
        if (finger === "Thumb" || n.lengthSq() < 0.5) continue;
        const segs = [sub(H(idx[1]), H(idx[0])), sub(H(idx[2]), H(idx[1])), sub(H(idx[3]), H(idx[2]))];
        if (segs.some((s) => s.length() < MIN_SEG)) continue;      // landmark noise: keep the previous finger
        // measured joints are kept when a hand can do them AND they reproduce how open the finger is; otherwise (a
        // noisy fist, a zig-zag finger) the finger is rebuilt from its openness alone
        const open = openness(segs, a.length(), finger);
        let dirs = null;
        if (plausibleFinger(segs, fwd, n, across)) {
          const s0 = limitKnuckle(segs[0], fwd, n, across);
          const s1 = hinge(s0, segs[1], n, PIP_MAX, across);
          dirs = [s0, s1, hinge(s1, segs[2], n, DIP_MAX, across)];
          if (Math.abs(modelOpenness(dirs) - open) > 0.2) dirs = null;
        }
        if (!dirs) {  // folded toward the avatar's own palm (the hand bone as posed), not the keypoints' palm normal:
          // in noisy frames that normal can flip while the hand keeps its smoothed orientation, and the curl then
          // closed toward the back of the hand (48 of 141 curl-path finger frames)
          const hq = this.node(`${side}Hand`).getWorldQuaternion(new THREE.Quaternion());
          const af = this.rest[`${side}HandA`].clone().applyQuaternion(hq), bf = this.rest[`${side}HandB`].clone().applyQuaternion(hq);
          const nH = (side === "left" ? new THREE.Vector3().crossVectors(af, bf) : new THREE.Vector3().crossVectors(bf, af)).normalize();
          const fH = af.clone().addScaledVector(nH, -af.dot(nH)).normalize();
          const acH = new THREE.Vector3().crossVectors(fH, nH).normalize();
          dirs = curlFinger(segs.map((d) => d.clone().applyQuaternion(this.toHand(n, fwd, nH, fH))), fH, nH, acH, open);
        }
        dirs.forEach((d, k) => this.aim(`${side}${finger}${SEGMENTS[finger][k]}`, d));
      }
      // The thumb (its own anatomy) keeps the earlier guard against folding toward the back of the hand.
      for (const [finger, idx] of Object.entries(FINGERS)) {
        if (finger !== "Thumb") continue;
        SEGMENTS[finger].forEach((seg, k) => {
          const boneName = `${side}${finger}${seg}`;
          let d = sub(H(idx[k + 1]), H(idx[k]));
          // If the segment vector is too short (landmark noise), keep the previous rotation.
          if (d.length() < MIN_SEG) return;
          // Clamp: project the segment onto the plane perpendicular to the "behind hand" direction
          // (i.e., toward the camera from the palm). This prevents fingers from bending backward.
          // "backward" means pointing away from the palm, toward the back of the hand.
          // The hands are mirror images: cross(a, b) points out of the palm on the left hand and out of the back on
          // the right, so the right hand takes cross(b, a). A component along -palmNormal is hyper-extension.
          // (With cross(a, b) on both hands, 58% of right-hand finger segments in the KArSL lexicon were "clamped",
          // i.e. every curl toward the palm was flattened, against 14% on the left.)
          const palmNormal = side === "left" ? new THREE.Vector3().crossVectors(a, b).normalize()
                                             : new THREE.Vector3().crossVectors(b, a).normalize();
          if (palmNormal.lengthSq() > 0.01) {
            const backAmount = d.dot(palmNormal);
            // Allow slight backward bend (sign lang fingers sometimes bend back slightly), but cap it.
            if (backAmount < -0.15 * d.length()) {
              // Remove the excessive backward component
              d = d.clone().addScaledVector(palmNormal, -backAmount - 0.15 * d.length());
            }
          }
          this.aim(boneName, d);
        });
      }
    }
    // a little head turn toward where the nose points, relative to the neck
    const nose = sub(P(NOSE), P(NECK));
    const head = this.node("head");
    if (head && nose.lengthSq() > 1e-8) {
      const yaw = THREE.MathUtils.clamp(Math.atan2(nose.x, 0.6) * 0.5, -0.35, 0.35);
      this.set("head", new THREE.Quaternion().setFromEuler(new THREE.Euler(0, yaw, 0)));
    }
  }

  // world positions of the bones the retargeting is judged by (scripts/avatars/retarget_eval.py)
  probe() {
    const out = {};
    for (const n of ["head", "neck", "leftEye", "rightEye", "leftUpperArm", "rightUpperArm", "leftLowerArm",
                     "rightLowerArm", "leftHand", "rightHand", "leftIndexDistal", "rightIndexDistal",
                     "leftMiddleDistal", "rightMiddleDistal", "leftMiddleProximal", "rightMiddleProximal",
                     "leftIndexProximal", "rightIndexProximal", "leftLittleProximal", "rightLittleProximal"]) {
      const b = this.node(n);
      out[n] = b ? b.getWorldPosition(new THREE.Vector3()).toArray() : null;
    }
    for (const side of ["left", "right"])                         // every finger joint, for finger checks
      for (const [f, segs] of Object.entries(SEGMENTS))
        for (const seg of segs) {
          const b = this.node(`${side}${f}${seg}`);
          out[`${side}${f}${seg}`] = b ? b.getWorldPosition(new THREE.Vector3()).toArray() : null;
        }
    out.reach = this.reachRatio ? [this.reachRatio.left ?? 0, this.reachRatio.right ?? 0] : null;
    // the face as rendered: ARKit shape weights on the mesh, or the VRM presets' values
    out.face = {};
    for (const [name, targets] of this.faceMorphs || []) out.face[name] = targets[0][0].morphTargetInfluences[targets[0][1]];
    for (const p of ["aa", "ou", "oh", "ee", "happy", "sad", "angry", "surprised", "blinkLeft", "blinkRight"]) {
      const v = this.vrm?.expressionManager?.getValue(p);
      if (v) out.face[`preset:${p}`] = v;
    }
    return out;
  }

  // show the frame for time t (seconds)
  render(t, dt = 1 / 30) {
    if (this.frames && this.frames.frames.length) {
      const i = Math.min(this.frames.frames.length - 1, Math.max(0, Math.round(t * this.frames.fps)));
      this.pose(this.frames.frames[i]);
      if (this.frames.blend) this.setFace(this.frames.blend[i], this.frames.blend_names);
      // the model's cloth (spring bones: the T-shirt's sleeves) starts where the T-pose left it and falls for several
      // frames after the arms snap down, flaring out at the sides of the first frames (and of a card's still preview):
      // let it settle on the first pose (about 2 s of simulation) before anything is drawn
      if (this.springReset && this.vrm) {
        for (let k = 0; k < 60; k++) this.vrm.update(1 / 30);
        this.springReset = false;
      }
    }
    this.vrm?.update(dt);
    if (this.faceMorphs?.size && this.face.size) this.applyFaceMorphs();
    this.renderer.render(this.scene, this.camera);
  }
}

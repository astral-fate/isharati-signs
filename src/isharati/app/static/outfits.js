// Modest outfits for any VRM avatar, built from simple shapes attached to its humanoid bones, so they follow the
// signing: a hijab (head cap open at the face + a drape over neck and shoulders) or a shemagh (checked headcloth
// with an agal), long sleeves on both arms, and clothing colours. The model's hair is hidden under head coverings.
// Sizes are measured from the avatar in its rest pose (face width, bone positions), so other VRM models fit too.
import * as THREE from "three";

export const PRESETS = {
  original: { style: "none" },
  hijab:    { style: "hijab",   head: "#2f3e5c", clothes: "#6d5a78", sleeves: true },
  shemagh:  { style: "shemagh", head: "#c1272d", clothes: "#f3f0e8", sleeves: true },
};

const toon = (color, map = null) =>
  new THREE.MeshToonMaterial({ color: new THREE.Color(color), map, side: THREE.DoubleSide });

function checks(red) {  // the red-and-white shemagh pattern
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const g = c.getContext("2d");
  g.fillStyle = "#f7f4ee";
  g.fillRect(0, 0, 128, 128);
  g.fillStyle = red;
  for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) if ((x + y) % 2 === 0) g.fillRect(x * 16 + 1, y * 16 + 1, 14, 14);
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(3, 2);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

export function applyOutfit(av, o) {
  const vrm = av.vrm;
  for (const m of av.outfit || []) m.parent?.remove(m);
  av.outfit = [];

  // hair and clothing colours on the model's own materials
  vrm.scene.traverse((obj) => {
    for (const m of [].concat(obj.material || [])) {
      m.userData.orig ||= { visible: m.visible, color: m.color?.clone(), shade: m.shadeColorFactor?.clone() };
      const name = m.name || "";
      m.visible = m.userData.orig.visible && !(o.style !== "none" && /HAIR/i.test(name));
      if (/Tops|Bottoms/i.test(name) && m.color) {
        const c = o.style === "none" || !o.clothes ? m.userData.orig.color : new THREE.Color(o.clothes);
        m.color.copy(c);
        if (m.shadeColorFactor) m.shadeColorFactor.copy(o.style === "none" ? m.userData.orig.shade : c.clone().multiplyScalar(0.72));
      }
    }
  });
  if (o.style === "none") return;

  // attach a piece to a bone, given where it should sit in the rest pose (world position and rotation)
  const attach = (bone, mesh, at, rot = new THREE.Quaternion()) => {
    const r = av.restBone[bone], n = av.node(bone);
    if (!r || !n) return;
    const inv = r.quat.clone().invert();
    mesh.position.copy(at.clone().sub(r.pos).applyQuaternion(inv));
    mesh.quaternion.copy(inv.clone().multiply(rot));
    n.add(mesh);
    av.outfit.push(mesh);
  };
  const B = (name) => av.restBone[name]?.pos.clone();
  const fb = av.faceBox;
  const faceW = fb ? fb.max.x - fb.min.x : 0.14;
  const faceC = fb ? fb.getCenter(new THREE.Vector3()) : B("head").add(new THREE.Vector3(0, 0.08, 0.02));
  const R = faceW * 0.62;                                   // head radius
  const headC = new THREE.Vector3(faceC.x, faceC.y + R * 0.18, faceC.z - R * 0.32);
  const chin = faceC.y - (fb ? (fb.max.y - fb.min.y) * 0.48 : R * 0.9);
  // outer shoulder width: the upper-arm joints sit inside the shoulders
  const shoulderW = (B("leftUpperArm") && B("rightUpperArm") ? B("leftUpperArm").distanceTo(B("rightUpperArm")) : R * 2) * 1.9;
  const chestY = (B("upperChest") || B("chest")).y;
  const headColor = o.head || "#2f3e5c";

  if (o.style === "hijab") {
    const m = toon(headColor);
    const cap = new THREE.Mesh(new THREE.SphereGeometry(R * 1.1, 48, 16, 0, Math.PI * 2, 0, Math.PI * 0.41), m);
    attach("head", cap, headC);
    const gap = Math.PI * 0.36;                             // the face opening, centred on the front (+Z)
    const band = new THREE.Mesh(new THREE.SphereGeometry(R * 1.1, 48, 24, Math.PI / 2 + gap / 2, Math.PI * 2 - gap,
                                                         Math.PI * 0.41 - 0.01, Math.PI * 0.39), m);
    attach("head", band, headC);
    // the drape: from under the chin over the neck and shoulders, a little flattened front to back
    // it ends on the shoulders, above the chest, so the body cannot poke through
    const top = R * 1.02, bottom = shoulderW * 0.5, h = Math.max(0.05, chin - R * 0.05 - (chestY + R * 0.15));
    // a soft falling profile (narrow at the neck, rounding out over the shoulders) instead of a straight cone
    const profile = [];
    for (let k = 0; k <= 12; k++) {
      const s = k / 12;
      profile.push(new THREE.Vector2(top + (bottom - top) * Math.sin(s * Math.PI / 2) ** 1.6, h / 2 - s * h));
    }
    const drape = new THREE.Mesh(new THREE.LatheGeometry(profile, 48), m);
    drape.scale.set(1, 1, 0.78);
    attach(av.restBone.upperChest ? "upperChest" : "chest", drape,
           new THREE.Vector3(headC.x, chin - R * 0.05 - h / 2, headC.z + R * 0.1));
  }

  if (o.style === "shemagh") {
    const tex = checks(headColor);
    const cloth = toon("#ffffff", tex);
    const cap = new THREE.Mesh(new THREE.SphereGeometry(R * 1.12, 48, 16, 0, Math.PI * 2, 0, Math.PI * 0.43), cloth);
    attach("head", cap, headC);
    // the cloth falls from the crown down both sides and the back to the shoulders, open at the face
    const gap = Math.PI * 0.42, h = headC.y + R * 0.2 - (chestY - R * 0.2);
    const fall = new THREE.Mesh(new THREE.CylinderGeometry(R * 1.12, shoulderW * 0.44, h, 48, 1, true, gap / 2, Math.PI * 2 - gap), cloth);
    fall.scale.set(1, 1, 0.9);
    attach("head", fall, new THREE.Vector3(headC.x, headC.y + R * 0.2 - h / 2, headC.z - R * 0.05));
    // the agal: two black cords around the crown
    const black = toon("#151515");
    for (const [dy, s] of [[R * 0.5, 1.0], [R * 0.62, 0.96]]) {
      const ring = new THREE.Mesh(new THREE.TorusGeometry(R * 1.08 * s, R * 0.07, 10, 48), black);
      attach("head", ring, headC.clone().add(new THREE.Vector3(0, dy, 0)),
             new THREE.Quaternion().setFromEuler(new THREE.Euler(Math.PI / 2 - 0.12, 0, 0)));
    }
  }

  if (o.sleeves) {
    const m = toon(o.clothes || "#6d5a78");
    const up = new THREE.Vector3(0, 1, 0);
    for (const side of ["left", "right"]) {
      for (const [a, b, r, len] of [[`${side}UpperArm`, `${side}LowerArm`, faceW * 0.3, 1.02],
                                    [`${side}LowerArm`, `${side}Hand`, faceW * 0.25, 0.96]]) {
        if (!B(a) || !B(b)) continue;
        const d = B(b).sub(B(a)), L = d.length() * len;
        const sleeve = new THREE.Mesh(new THREE.CylinderGeometry(r * 0.92, r, L, 20, 1, true), m);
        attach(a, sleeve, B(a).add(d.clone().normalize().multiplyScalar(L / 2)),
               new THREE.Quaternion().setFromUnitVectors(up, d.clone().normalize()));
      }
    }
  }
}

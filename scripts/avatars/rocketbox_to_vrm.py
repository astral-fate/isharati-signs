"""Convert a Microsoft Rocketbox avatar (MIT; github.com/microsoft/Microsoft-Rocketbox) to VRM 1.0 for the web app.

Run inside Blender with the VRM Add-on for Blender (MIT) installed:
  blender --background --python scripts/avatars/rocketbox_to_vrm.py -- <avatar dir> <out.vrm> [texture px]

<avatar dir> is a Rocketbox Assets/Avatars/... folder (Export/<name>_facial.fbx or <name>.fbx, Textures/*.tga).
Steps: import the FBX; re-link its textures from the avatar's Textures folder and downscale them (2048 -> 1024 px by
default; the specular maps are dropped, glTF has no slot for them); map the 3ds Max Biped bones to the VRM humanoid,
including all finger segments; keep the A-pose rest; fill the VRM metadata (Rocketbox, MIT); export.
The app's retargeting (src/isharati/app/static/avatar.js) then drives it like any other VRM.
"""
import sys
from pathlib import Path

import bpy

args = [a for a in sys.argv[sys.argv.index("--") + 1:] if not a.startswith("--")]  # positional; flags read below
SRC, OUT = Path(args[0]).resolve(), Path(args[1]).resolve()
TEX_PX = int(args[2]) if len(args) > 2 else 1024
NAME = SRC.name

B = "Bip01 "
SIDES = {"left": "L", "right": "R"}
FINGERS = {"thumb": ("Finger0", ("metacarpal", "proximal", "distal")),
           "index": ("Finger1", ("proximal", "intermediate", "distal")),
           "middle": ("Finger2", ("proximal", "intermediate", "distal")),
           "ring": ("Finger3", ("proximal", "intermediate", "distal")),
           "little": ("Finger4", ("proximal", "intermediate", "distal"))}
MAP = {"hips": "Pelvis", "spine": "Spine", "chest": "Spine1", "upper_chest": "Spine2", "neck": "Neck", "head": "Head",
       "left_eye": "LEye", "right_eye": "REye", "jaw": "MJaw"}
for side, s in SIDES.items():
    MAP.update({f"{side}_shoulder": f"{s} Clavicle", f"{side}_upper_arm": f"{s} UpperArm", f"{side}_lower_arm": f"{s} Forearm",
                f"{side}_hand": f"{s} Hand", f"{side}_upper_leg": f"{s} Thigh", f"{side}_lower_leg": f"{s} Calf",
                f"{side}_foot": f"{s} Foot", f"{side}_toes": f"{s} Toe0"})
    for finger, (base, segs) in FINGERS.items():   # Biped: Finger0, Finger01, Finger02; Finger1, Finger11, Finger12 ...
        d = base[-1]
        for k, seg in enumerate(segs):
            MAP[f"{side}_{finger}_{seg}"] = f"{s} Finger{d}" + ("" if k == 0 else str(k))


def fbx_path():
    # --facial: the version with 176 blendshapes, of which keep_face_shapes() keeps the ARKit 52
    for n in ((f"{NAME}_facial.fbx", f"{NAME}.fbx") if "--facial" in sys.argv else (f"{NAME}.fbx", f"{NAME}_facial.fbx")):
        p = SRC / "Export" / n
        if p.exists():
            return p
    raise SystemExit(f"no FBX in {SRC / 'Export'}")


def fix_textures(tmp: Path):
    """Point every image at the local Textures folder, downscaled, saved as PNG; drop specular maps."""
    tmp.mkdir(parents=True, exist_ok=True)
    for mat in bpy.data.materials:
        if not mat.use_nodes:
            continue
        nt = mat.node_tree
        for node in list(nt.nodes):
            if node.type != "TEX_IMAGE" or node.image is None:
                continue
            fname = Path(node.image.filepath.replace("\\", "/")).name
            local = SRC / "Textures" / fname
            if "specular" in fname.lower() or not local.exists():
                nt.nodes.remove(node)
                continue
            img = bpy.data.images.load(str(local), check_existing=True)
            if max(img.size) > TEX_PX:
                img.scale(TEX_PX, TEX_PX)
            out = tmp / (Path(fname).stem + ".png")
            img.filepath_raw = str(out)
            img.file_format = "PNG"
            img.save()
            node.image = img
        if "opacity" in mat.name.lower():  # hair, lashes: alpha from the colour texture
            mat.blend_method = "HASHED" if hasattr(mat, "blend_method") else mat.blend_method
            bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
            tex = next((n for n in nt.nodes if n.type == "TEX_IMAGE"), None)
            if bsdf and tex:
                nt.links.new(tex.outputs["Alpha"], bsdf.inputs["Alpha"])


# The face. Rocketbox's facial FBX carries ARKit's 52 blendshapes (AK_01_BrowDownLeft ... AK_52_TongueOut), the same
# set MediaPipe's face landmarker scores, so the app drives them one to one from the signer's face
# (scripts/face/attach.py -> avatar.js setFace). Everything else (visemes AA_VI_*, FACS units AU_*, HB_*, SR_*:
# 124 shapes) is dropped: glTF stores each shape over the whole body mesh, ~0.1 MB apiece.
KEEP_SHAPES = ("Basis", "AK_")
# VRM 1.0 presets -> (shape key, weight): lets any VRM player blink, speak and emote with this avatar
PRESETS = {
    "blink": [("AK_09_EyeBlinkLeft", 1), ("AK_10_EyeBlinkRight", 1)],
    "blink_left": [("AK_09_EyeBlinkLeft", 1)], "blink_right": [("AK_10_EyeBlinkRight", 1)],
    "aa": [("AK_25_JawOpen", 1)], "ih": [("AK_25_JawOpen", 0.3), ("AK_46_MouthStretchLeft", 0.6),
                                        ("AK_47_MouthStretchRight", 0.6)],
    "ou": [("AK_38_MouthPucker", 1), ("AK_25_JawOpen", 0.2)], "ee": [("AK_46_MouthStretchLeft", 1),
                                                                     ("AK_47_MouthStretchRight", 1)],
    "oh": [("AK_32_MouthFunnel", 1), ("AK_25_JawOpen", 0.4)],
    "happy": [("AK_44_MouthSmileLeft", 1), ("AK_45_MouthSmileRight", 1), ("AK_07_CheekSquintLeft", 0.5),
              ("AK_08_CheekSquintRight", 0.5)],
    "sad": [("AK_30_MouthFrownLeft", 1), ("AK_31_MouthFrownRight", 1), ("AK_03_BrowInnerUp", 0.8)],
    "angry": [("AK_01_BrowDownLeft", 1), ("AK_02_BrowDownRight", 1), ("AK_36_MouthPressLeft", 0.5),
              ("AK_37_MouthPressRight", 0.5)],
    "surprised": [("AK_04_BrowOuterUpLeft", 1), ("AK_05_BrowOuterUpRight", 1), ("AK_03_BrowInnerUp", 1),
                  ("AK_21_EyeWideLeft", 1), ("AK_22_EyeWideRight", 1), ("AK_25_JawOpen", 0.3)],
    "relaxed": [("AK_44_MouthSmileLeft", 0.3), ("AK_45_MouthSmileRight", 0.3)],
}


def keep_face_shapes():
    """Drop every shape key but Basis and the ARKit 52; return the mesh that has them (None without a face)."""
    face = None
    for o in bpy.data.objects:
        if o.type != "MESH" or not o.data.shape_keys:
            continue
        o.active_shape_key_index = 0
        for k in list(o.data.shape_keys.key_blocks):
            if not k.name.startswith(KEEP_SHAPES):
                o.shape_key_remove(k)
        if any(k.name.startswith("AK_") for k in o.data.shape_keys.key_blocks):
            face = o
    return face


def bind_presets(ext, face):
    keys = {k.name for k in face.data.shape_keys.key_blocks}
    for preset, binds in PRESETS.items():
        expr = getattr(ext.vrm1.expressions.preset, preset)
        expr.morph_target_binds.clear()
        for key, weight in binds:
            if key in keys:
                b = expr.morph_target_binds.add()
                b.node.mesh_object_name = face.name
                b.index = key
                b.weight = weight


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.preferences.addon_enable(module="bl_ext.user_default.vrm")
    bpy.ops.import_scene.fbx(filepath=str(fbx_path()), automatic_bone_orientation=False)
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    # Biped parents the thighs to the spine and the clavicles to the neck; VRM needs the legs under the hips and the
    # shoulders under the upper chest (re-parenting keeps the skinning unchanged)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm.data.edit_bones
    for s in ("L", "R"):
        eb[f"{B}{s} Thigh"].parent = eb[f"{B}Pelvis"]
        eb[f"{B}{s} Clavicle"].parent = eb[f"{B}Spine2"]
    bpy.ops.object.mode_set(mode="OBJECT")

    # metres, no object scale (the FBX is in centimetres with a 0.01 armature scale). Done after the edit-mode
    # step: leaving edit mode restores the object scale while keeping applied bone lengths, which shrinks the mesh
    for o in bpy.data.objects:
        o.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    fix_textures(OUT.parent / f".{NAME}_textures")
    face = keep_face_shapes()
    ext = arm.data.vrm_addon_extension
    ext.spec_version = "1.0"
    bpy.ops.vrm.assign_vrm1_humanoid_human_bones_automatically(armature_object_name=arm.name)
    hb = ext.vrm1.humanoid.human_bones
    names = {b.name for b in arm.data.bones}
    missing = []
    for prop, bone in MAP.items():
        full = B + bone
        if full in names and hasattr(hb, prop):
            getattr(hb, prop).node.bone_name = full
        elif prop not in ("jaw", "left_eye", "right_eye", "left_toes", "right_toes", "upper_chest"):
            missing.append((prop, full))
    if missing:
        print("MISSING", missing)

    # the rest pose stays the Rocketbox A-pose: the app measures each bone's rest direction from its position, and
    # baking a T-pose into a mesh with 176 shape keys is not possible without losing them

    if face is not None:
        bind_presets(ext, face)
        print("FACE", face.name, len(face.data.shape_keys.key_blocks) - 1, "shape keys")

    meta = ext.vrm1.meta
    meta.vrm_name = f"Rocketbox {NAME}"
    meta.authors.clear()
    meta.authors.add().value = "Microsoft Rocketbox"
    meta.copyright_information = "Microsoft Rocketbox Avatar Library, MIT License"
    meta.license_url = "https://vrm.dev/licenses/1.0/"
    meta.other_license_url = "https://github.com/microsoft/Microsoft-Rocketbox/blob/master/LICENSE.md"
    meta.allow_redistribution = True
    meta.modification = "allowModificationRedistribution"
    meta.commercial_usage = "corporation"
    meta.allow_political_or_religious_usage = True
    meta.credit_notation = "required"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    import importlib
    validation = importlib.import_module("bl_ext.user_default.vrm.editor.validation")
    original = validation.WM_OT_vrm_validator.set_error_collection

    def report(collection, state):  # show the add-on's validation messages (it only shows them in a dialog)
        for m in state.error_messages:
            print("VRM ERROR:", m)
        return original(collection, state)
    validation.WM_OT_vrm_validator.set_error_collection = staticmethod(report)
    r = bpy.ops.export_scene.vrm(filepath=str(OUT), armature_object_name=arm.name, ignore_warning=True)
    print("EXPORT", r, OUT, round(OUT.stat().st_size / 1e6, 1) if OUT.exists() else None, "MB")


main()

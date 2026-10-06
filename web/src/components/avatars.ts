export const DEFAULT_AVATAR = "rocketbox_female_06.vrm";

export const AVATAR_KEYS: Record<string, string> = {
  "rocketbox_female_06.vrm": "avatar.female_06",
  "rocketbox_female_10.vrm": "avatar.female_10",
  "rocketbox_male_15.vrm": "avatar.male_15",
  "rocketbox_male_19.vrm": "avatar.male_19",
  "rocketbox_male_21.vrm": "avatar.male_21",
  "avatar.vrm": "avatar.stylised",
};

export function pickAvatar(saved: string | null, available: string[]): string {
  if (saved && available.includes(saved)) return saved;
  if (available.includes(DEFAULT_AVATAR)) return DEFAULT_AVATAR;
  return available[0] ?? DEFAULT_AVATAR;
}

// Two styles to choose between: the cartoon (pixiv VRM sample) and the realistic Microsoft Rocketbox avatars
// converted in Blender. Same signs, same retargeting; only the model changes.
export type AvatarStyle = "cartoon" | "realistic";
export type Outfit = "hijab" | "shemagh" | "original";
export const STYLES: AvatarStyle[] = ["cartoon", "realistic"];
export const OUTFITS: Outfit[] = ["hijab", "shemagh", "original"];
export const styleOf = (model: string): AvatarStyle => (model.startsWith("rocketbox_") ? "realistic" : "cartoon");
export const avatarsFor = (style: AvatarStyle, available: string[]) => available.filter((m) => styleOf(m) === style);

export function avatarForStyle(style: AvatarStyle, last: Partial<Record<AvatarStyle, string>>, available: string[]) {
  const pool = avatarsFor(style, available);
  const remembered = last[style];
  if (remembered && pool.includes(remembered)) return remembered;
  if (pool.includes(DEFAULT_AVATAR)) return DEFAULT_AVATAR;
  return pool[0] ?? pickAvatar(null, available);
}

// same presets as the classic page (static/outfits.js PRESETS); Rocketbox avatars come dressed modestly
const PRESET = {
  original: { style: "none" },
  hijab: { style: "hijab", head: "#2f3e5c", clothes: "#6d5a78", sleeves: true },
  shemagh: { style: "shemagh", head: "#c1272d", clothes: "#f3f0e8", sleeves: true },
};
export const outfitFor = (model: string, outfit: Outfit = "hijab") =>
  styleOf(model) === "realistic" ? PRESET.original : PRESET[outfit];

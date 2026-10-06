import { AVATAR_KEYS, DEFAULT_AVATAR, avatarForStyle, avatarsFor, outfitFor, pickAvatar, styleOf } from "./avatars";

const all = Object.keys(AVATAR_KEYS);

test("a saved avatar is kept when it still exists", () => {
  expect(pickAvatar("rocketbox_male_19.vrm", all)).toBe("rocketbox_male_19.vrm");
});

test("a saved avatar that no longer exists falls back to the default", () => {
  expect(pickAvatar("old_model.vrm", all)).toBe(DEFAULT_AVATAR);
  expect(pickAvatar(null, all)).toBe(DEFAULT_AVATAR);
});

test("without the default, the first available avatar is used", () => {
  expect(pickAvatar(null, ["avatar.vrm"])).toBe("avatar.vrm");
});

test("realistic avatars keep their own clothes; the cartoon one wears the chosen outfit (hijab by default)", () => {
  expect(outfitFor("rocketbox_female_06.vrm", "shemagh")).toEqual({ style: "none" });
  expect(outfitFor("avatar.vrm")).toMatchObject({ style: "hijab" });
  expect(outfitFor("avatar.vrm", "shemagh")).toMatchObject({ style: "shemagh" });
  expect(outfitFor("avatar.vrm", "original")).toEqual({ style: "none" });
});

test("styles split the avatars into cartoon and realistic", () => {
  expect(styleOf("avatar.vrm")).toBe("cartoon");
  expect(styleOf("rocketbox_male_19.vrm")).toBe("realistic");
  expect(avatarsFor("realistic", all)).toHaveLength(5);
  expect(avatarsFor("cartoon", all)).toEqual(["avatar.vrm"]);
});

test("switching style returns to the last avatar used in that style", () => {
  const last = { cartoon: "avatar.vrm", realistic: "rocketbox_male_21.vrm" };
  expect(avatarForStyle("realistic", last, all)).toBe("rocketbox_male_21.vrm");
  expect(avatarForStyle("realistic", { cartoon: "avatar.vrm", realistic: "gone.vrm" }, all)).toBe(DEFAULT_AVATAR);
  expect(avatarForStyle("cartoon", { realistic: DEFAULT_AVATAR }, all)).toBe("avatar.vrm");
});

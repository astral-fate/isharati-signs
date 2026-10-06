import { ApiError } from "../api";
import { EXAMPLES, SIGN_LANGS, errorKey, textDir } from "./signLanguages";

test("text direction follows the sign language, not the interface", () => {
  expect(textDir("en")).toBe("ltr");
  expect(textDir("tr")).toBe("ltr");
  expect(textDir("ar")).toBe("rtl");
  expect(textDir("ur")).toBe("rtl");
});

test("four sign languages, each with examples", () => {
  expect(SIGN_LANGS.map((s) => s.code)).toEqual(["en", "ar", "tr", "ur"]);
  for (const s of SIGN_LANGS) expect(EXAMPLES[s.code].length).toBeGreaterThanOrEqual(3);
});

test("errors map to localised messages", () => {
  expect(errorKey(new ApiError("html", "x"))).toBe("try.html");
  expect(errorKey(new ApiError("quota", "x"))).toBe("try.quota");
  expect(errorKey(new ApiError("quota", "x"), "text")).toBe("try.quotaStudio");
  expect(errorKey(new ApiError("server", "x"))).toBe("try.error");
  expect(errorKey(new Error("x"))).toBe("try.error");
});

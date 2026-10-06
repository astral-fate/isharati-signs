import { ApiError, ask, getClips, signText, transcribeAudio } from "./api";

const reply = (body: string, status: number, type: string) =>
  vi.fn().mockResolvedValue(new Response(body, { status, headers: { "content-type": type } }));

afterEach(() => vi.unstubAllGlobals());

test("an HTML page instead of JSON is an 'html' error", async () => {
  vi.stubGlobal("fetch", reply("<!DOCTYPE html><p>sign in</p>", 200, "text/html"));
  await expect(ask("q", "en")).rejects.toMatchObject({ kind: "html" });
});

test("the free-quota 503 is a 'quota' error", async () => {
  vi.stubGlobal("fetch", reply(JSON.stringify({ detail: "quota used up" }), 503, "application/json"));
  const err = await ask("q", "en").catch((e) => e);
  expect(err).toMatchObject({ kind: "quota" });
  expect(err.message).toContain("quota used up");
});

test("other failures carry the server's detail", async () => {
  vi.stubGlobal("fetch", reply(JSON.stringify({ detail: "GlossError: boom" }), 500, "application/json"));
  const err = await ask("q", "tr").catch((e) => e);
  expect(err).toBeInstanceOf(ApiError);
  expect(err.kind).toBe("server");
  expect(err.message).toContain("boom");
});

test("a network failure is a 'network' error", async () => {
  vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
  await expect(ask("q", "ur")).rejects.toMatchObject({ kind: "network" });
});

test("a signed report is returned as is", async () => {
  vi.stubGlobal("fetch", reply(JSON.stringify({ status: "signed", id: "abc" }), 200, "application/json"));
  await expect(ask("q", "ar")).resolves.toMatchObject({ status: "signed", id: "abc" });
});

test("signing a text posts it to /api/sign, not /api/ask", async () => {
  const f = reply(JSON.stringify({ status: "signed", mode: "text", id: "tenabc" }), 200, "application/json");
  vi.stubGlobal("fetch", f);
  await expect(signText("Prayer is light.", "en")).resolves.toMatchObject({ mode: "text" });
  expect(f.mock.calls[0][0]).toBe("/api/sign");
  expect(JSON.parse(f.mock.calls[0][1].body)).toEqual({ text: "Prayer is light.", lang: "en" });
});

test("missing clips give an empty list", async () => {
  vi.stubGlobal("fetch", reply("Not Found", 404, "text/plain"));
  await expect(getClips()).resolves.toEqual([]);
});

test("a 200 reply with unparseable JSON body rejects with kind server", async () => {
  vi.stubGlobal("fetch", reply("{not json", 200, "application/json"));
  await expect(ask("q", "en")).rejects.toMatchObject({ kind: "server" });
});

test("a 422 with detail as array rejects with kind server and message containing error", async () => {
  vi.stubGlobal("fetch", reply(JSON.stringify({ detail: [{ msg: "field required" }] }), 422, "application/json"));
  const err = await ask("q", "en").catch((e) => e);
  expect(err).toMatchObject({ kind: "server" });
  expect(err.message).toContain("field required");
});

test("getClips with a 200 JSON object (not array) resolves to empty list", async () => {
  vi.stubGlobal("fetch", reply(JSON.stringify({}), 200, "application/json"));
  await expect(getClips()).resolves.toEqual([]);
});

test("transcribeAudio sends multipart form data to /api/transcribe", async () => {
  const f = reply(JSON.stringify({ text: "الحمد لله", segments: [{ text: "الحمد لله", start: 0, end: 1.5 }] }), 200, "application/json");
  vi.stubGlobal("fetch", f);
  const blob = new Blob(["test audio content"], { type: "audio/wav" });
  const res = await transcribeAudio(blob);
  expect(res.text).toBe("الحمد لله");
  expect(res.segments).toHaveLength(1);
  expect(f.mock.calls[0][0]).toBe("/api/transcribe");
  expect(f.mock.calls[0][1].method).toBe("POST");
  expect(f.mock.calls[0][1].body).toBeInstanceOf(FormData);
});


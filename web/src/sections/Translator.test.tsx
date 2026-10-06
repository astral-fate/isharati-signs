import * as rtl from "@testing-library/react";
const { act, fireEvent, screen } = rtl;
import en from "../i18n/en.json";
import { I18nProvider } from "../i18n/i18n";
import { AppStateProvider } from "../state";
import type { Report } from "../types";
import { Translator } from "./Translator";

vi.mock("../api", async (orig) => ({ ...(await orig<typeof import("../api")>()), ask: vi.fn(), signText: vi.fn(), getPose: vi.fn() }));
vi.mock("../components/AvatarView", () => ({ AvatarView: () => <canvas aria-label="avatar" /> }));
vi.mock("../components/KeypointCanvas", () => ({ KeypointCanvas: () => <canvas aria-label="keypoints" /> }));
import * as api from "../api";

const ask = vi.mocked(api.ask), getPose = vi.mocked(api.getPose);
const report = (answer: string): Report => ({ status: "signed", id: answer, answer, sources: [], segments: [], coverage: 1 });
function deferred<T>() {
  let resolve!: (v: T) => void;
  const promise = new Promise<T>((r) => (resolve = r));
  return { promise, resolve };
}
const flush = () => act(async () => { await Promise.resolve(); });
const submit = (text: string) => {
  fireEvent.change(screen.getByRole("textbox"), { target: { value: text } });
  fireEvent.submit(screen.getByRole("textbox").closest("form")!);
};

beforeEach(() => {
  localStorage.clear();
  history.replaceState(null, "", "/");
  ask.mockReset(); getPose.mockReset();
  getPose.mockResolvedValue({ fps: 30, frames: [[[0, 0, 0]]] });
  rtl.render(<I18nProvider initial="en"><AppStateProvider><Translator /></AppStateProvider></I18nProvider>);
});

test("the chosen sign language stays after a result and while the next question is pending", async () => {
  const pending = deferred<Report>();
  ask.mockResolvedValueOnce(report("one")).mockReturnValueOnce(pending.promise);
  fireEvent.click(screen.getByText("Türkçe → TİD"));
  fireEvent.click(screen.getByText("Zekât nedir?"));
  await flush(); await flush();
  expect(screen.getByText("one")).toBeInTheDocument();
  fireEvent.click(screen.getByText("Hac nedir?"));            // second question, still pending
  expect(screen.getByRole("tab", { name: "Türkçe → TİD" })).toHaveAttribute("aria-selected", "true");
  const input = screen.getByRole("textbox");
  expect(input).toHaveAttribute("dir", "ltr");
  expect(input).toHaveAttribute("lang", "tr");
  expect(ask.mock.calls.map((c) => c[1])).toEqual(["tr", "tr"]);
  pending.resolve(report("two"));
  await flush(); await flush();
  fireEvent.click(screen.getByText("العربية → ArSL"));         // after a result, the tab choice sticks
  expect(screen.getByRole("tab", { name: "العربية → ArSL" })).toHaveAttribute("aria-selected", "true");
  expect(screen.getByRole("textbox")).toHaveAttribute("lang", "ar");
});

test("a slow first answer does not replace a newer one", async () => {
  const slow = deferred<Report>();
  ask.mockReturnValueOnce(slow.promise).mockResolvedValueOnce(report("second"));
  submit("first");
  submit("second");
  await flush(); await flush();
  // the answer panel (the question box is a textarea now, which holds the typed text too)
  expect(screen.getByText("second", { selector: ".answer" })).toBeInTheDocument();
  slow.resolve(report("first"));
  await flush(); await flush();
  expect(screen.queryByText("first", { selector: ".answer" })).toBeNull();
  expect(screen.getByText("second", { selector: ".answer" })).toBeInTheDocument();
});

test("switching tab while pending discards the late answer", async () => {
  const slow = deferred<Report>();
  ask.mockReturnValueOnce(slow.promise);
  submit("late");
  fireEvent.click(screen.getByText("العربية → ArSL"));
  slow.resolve(report("late"));
  await flush(); await flush();
  expect(screen.queryByText("late")).toBeNull();
  expect(screen.queryByText(en["try.wait"])).toBeNull();
});

test("a pose that fails to load is reported", async () => {
  ask.mockResolvedValueOnce(report("ok"));
  getPose.mockRejectedValueOnce(new Error("404"));
  submit("ok");
  await flush(); await flush(); await flush();
  expect(screen.getByText(en["try.noPose"])).toBeInTheDocument();
});

test("a source links only when its url is http(s)", async () => {
  ask.mockResolvedValueOnce({ ...report("cited"), sources: [
    { reference: "Good", url: "https://example.org/a" }, { reference: "Bad", url: "javascript:alert(1)" }] } as Report);
  submit("cited");
  await flush(); await flush();
  expect(screen.getByText("Good").closest("a")).toHaveAttribute("href", "https://example.org/a");
  expect(screen.getByText("Bad").closest("a")).toBeNull();
});

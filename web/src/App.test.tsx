import { act, render, screen } from "@testing-library/react";
import App from "./App";
import en from "./i18n/en.json";
import { I18nProvider } from "./i18n/i18n";

beforeEach(() => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("nope", { status: 404, headers: { "content-type": "text/plain" } })));
  HTMLCanvasElement.prototype.getContext = vi.fn(() => null) as any;
});
afterEach(() => vi.unstubAllGlobals());

test("the page renders every section, with built-in figures when the API is unreachable", async () => {
  await act(async () => { render(<I18nProvider initial="en"><App /></I18nProvider>); });
  for (const key of ["hero.title", "try.title", "how.title", "lex.title", "av.title", "src.title"]) {
    expect(screen.getAllByText(en[key as keyof typeof en]).length).toBeGreaterThan(0);
  }
  for (const id of ["try", "how", "lexicon", "avatars", "sources"]) expect(document.getElementById(id)).not.toBeNull();
  expect(screen.getAllByText(/TİD/).length).toBeGreaterThan(0);
  expect(screen.getAllByText(/ISL/).length).toBeGreaterThan(0);
});

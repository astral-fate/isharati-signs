import { act, render, screen } from "@testing-library/react";
import en from "../i18n/en.json";
import { I18nProvider } from "../i18n/i18n";
import { AppStateProvider, useAppState } from "../state";
import { StyleSwitch } from "./StyleSwitch";

function Current() { const { avatar, outfit } = useAppState(); return <p data-testid="cur">{avatar}|{outfit}</p>; }

beforeEach(() => {
  localStorage.clear();
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("x", { status: 404, headers: { "content-type": "text/plain" } })));
});
afterEach(() => vi.unstubAllGlobals());

test("switching style swaps the avatar and back, remembering the realistic choice", async () => {
  await act(async () => {
    render(<I18nProvider initial="en"><AppStateProvider><StyleSwitch /><Current /></AppStateProvider></I18nProvider>);
  });
  expect(screen.getByTestId("cur").textContent).toBe("rocketbox_female_06.vrm|hijab");  // realistic by default
  act(() => screen.getByTitle(en["avatar.male_19"]).click());
  act(() => screen.getByText(en["style.cartoon"]).click());
  expect(screen.getByTestId("cur").textContent).toBe("avatar.vrm|hijab");
  act(() => screen.getByText(en["outfit.shemagh"]).click());
  expect(screen.getByTestId("cur").textContent).toBe("avatar.vrm|shemagh");
  act(() => screen.getByText(en["style.realistic"]).click());
  expect(screen.getByTestId("cur").textContent).toBe("rocketbox_male_19.vrm|shemagh");
});

import { act, render, screen } from "@testing-library/react";
import ar from "./ar.json";
import en from "./en.json";
import tr from "./tr.json";
import ur from "./ur.json";
import { I18nProvider, useI18n } from "./i18n";

function Probe() {
  const { t, setUi, dir } = useI18n();
  return <div><p>{t("hero.cta")}</p><p data-testid="dir">{dir}</p><p>{t("no.such.key")}</p>
    <button onClick={() => setUi("ar")}>ar</button><button onClick={() => setUi("tr")}>tr</button><button onClick={() => setUi("ur")}>ur</button></div>;
}

test("every language has every English key", () => {
  for (const [name, strings] of Object.entries({ ar, tr, ur })) {
    const missing = Object.keys(en).filter((k) => !(k in strings));
    expect(missing, name).toEqual([]);
  }
  for (const [name, strings] of Object.entries({ en, ar, tr, ur })) {
    const empty = Object.entries(strings).filter(([, v]) => !String(v).trim()).map(([k]) => k);
    expect(empty, `${name} empty strings`).toEqual([]);
  }
});

test("switching to Arabic sets right-to-left on the document", () => {
  render(<I18nProvider initial="en"><Probe /></I18nProvider>);
  expect(screen.getByText(en["hero.cta"])).toBeInTheDocument();
  act(() => screen.getByText("ar").click());
  expect(screen.getByText(ar["hero.cta"])).toBeInTheDocument();
  expect(document.documentElement.dir).toBe("rtl");
  expect(document.documentElement.lang).toBe("ar");
  act(() => screen.getByText("tr").click());
  expect(document.documentElement.dir).toBe("ltr");
  expect(screen.getByText("no.such.key")).toBeInTheDocument();   // unknown keys show the key, not a crash
  act(() => screen.getByText("ur").click());
  expect(document.documentElement.dir).toBe("rtl");
  expect(document.documentElement.lang).toBe("ur");
  expect(screen.getByText(ur["hero.cta"])).toBeInTheDocument();
});

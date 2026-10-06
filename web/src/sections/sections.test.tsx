import * as rtl from "@testing-library/react";
import App from "../App";
import { I18nProvider } from "../i18n/i18n";

const json = (body: unknown) => new Response(JSON.stringify(body), { status: 200, headers: { "content-type": "application/json" } });
const nope = () => new Response("nope", { status: 404, headers: { "content-type": "text/plain" } });

function stubFetch(routes: Record<string, unknown>) {
  vi.stubGlobal("fetch", vi.fn(async (url: string) => (url in routes ? json(routes[url]) : nope())));
}
async function mount() {
  await rtl.act(async () => { rtl.render(<I18nProvider initial="en"><App /></I18nProvider>); });
}

beforeEach(() => { HTMLCanvasElement.prototype.getContext = vi.fn(() => null) as any; });
afterEach(() => { rtl.cleanup(); vi.unstubAllGlobals(); });

test("switching the interface to Arabic and Urdu through the nav flips the page to rtl", async () => {
  stubFetch({});
  await mount();
  expect(document.documentElement.dir).toBe("ltr");
  const group = rtl.screen.getByRole("group", { name: "Interface language" });
  await rtl.act(async () => { rtl.fireEvent.click(rtl.within(group).getByText("ع")); });
  expect(document.documentElement.dir).toBe("rtl");
  await rtl.act(async () => { rtl.fireEvent.click(rtl.within(group).getByText("EN")); });
  expect(document.documentElement.dir).toBe("ltr");
  await rtl.act(async () => { rtl.fireEvent.click(rtl.within(group).getByText("اردو")); });
  expect(document.documentElement.dir).toBe("rtl");
  expect(document.documentElement.lang).toBe("ur");
});

test("the hero and the lexicon show the figures the API returns", async () => {
  stubFetch({
    "/api/stats": {
      languages: [{ code: "en", name: "English", sign_language: "ASL", signs: 1234, passages: 5678, coverage: 0.5 }],
      distinct_passages: 4321,
      licences: [{ source: "X", licence: "Y", use: "Z" }],
    },
  });
  // counters animate once in view: report every element as visible, then let the 1.6 s count-up finish
  vi.stubGlobal("IntersectionObserver", class {
    constructor(private cb: IntersectionObserverCallback) {}
    observe(el: Element) { this.cb([{ isIntersecting: true, target: el } as IntersectionObserverEntry], this as any); }
    unobserve() {} disconnect() {} takeRecords() { return []; }
  });
  await mount();
  const opts = { timeout: 6000 };
  await rtl.waitFor(() => expect(rtl.screen.getAllByText("1,234").length).toBeGreaterThanOrEqual(2), opts); // hero total + lexicon card
  await rtl.waitFor(() => {
    expect(rtl.screen.getAllByText("4,321").length).toBeGreaterThanOrEqual(1);   // hero: distinct passages
    expect(rtl.screen.getAllByText("5,678").length).toBeGreaterThanOrEqual(1);   // lexicon: this language's corpus
  }, opts);
});

test("recorded clips fill the avatar gallery and the hero reel", async () => {
  const clip = (avatar: string) => ({
    avatar, lang: "en", sign_language: "ASL", glosses: ["PRAY"],
    webm: `/clips/${avatar}.webm`, mp4: `/clips/${avatar}.mp4`, poster: `/clips/${avatar}.jpg`, seconds: 3,
  });
  stubFetch({ "/clips/clips.json": [clip("avatar.vrm"), clip("rocketbox_female_06.vrm")] });
  await mount();
  await rtl.waitFor(() => expect(document.querySelectorAll(".reel video").length).toBe(2));
  expect(document.querySelectorAll("#avatars .clip video").length).toBe(2);
  expect(document.querySelectorAll("#avatars .clip .poster").length).toBeGreaterThan(0); // the other avatars have none yet
});

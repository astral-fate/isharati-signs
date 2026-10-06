import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as api from "../api";
import { Hadith } from "./Hadith";

vi.mock("../api");
vi.mock("../i18n/i18n", () => ({ useI18n: () => ({ t: (k: string) => k, ui: "en" }) }));
vi.mock("../state", () => ({
  useAppState: () => ({ avatar: "cartoon", outfit: "hijab" }),
}));
vi.mock("../components/AvatarView", () => ({ AvatarView: () => <canvas data-testid="avatar" /> }));
vi.mock("../components/KeypointCanvas", () => ({ KeypointCanvas: () => <canvas data-testid="kp" /> }));
vi.mock("../components/StyleSwitch", () => ({ StyleSwitch: () => null }));
vi.mock("../components/Counter", () => ({ Counter: ({ value }: any) => <span>{value}</span> }));
vi.mock("../player", () => ({ usePlayer: () => ({ t: 0, playing: false, toggle: vi.fn(), seek: vi.fn(), play: vi.fn(), speed: 1, setSpeed: vi.fn() }) }));
vi.mock("framer-motion", () => ({
  motion: { div: ({ children, ...p }: any) => <div {...p}>{children}</div> },
  AnimatePresence: ({ children }: any) => children,
}));

const coverage = {
  samples: 3, hadith: 2, hours: 0.1, skipped: {},
  sources: [{ key: "eray_40hadis", name: "Eray Demir", sign_language: "TİD", samples: 2, hadith: 2, hours: 0.1 }],
  sign_languages: [{ sign_language: "ArSL", samples: 1, hadith: 1 }, { sign_language: "TİD", samples: 2, hadith: 2 }],
  collections: [{ collection: "nawawi", name: "الأربعون النووية", samples: 1, hadith: 1 },
                { collection: "muslim", name: "صحيح مسلم", samples: 2, hadith: 1 }],
};
const listItem = (ref: string) => ({
  ref, collection: ref.split(":")[0], collection_name: "", number: ref.split(":")[1], text: `matn ${ref}`,
  samples: 1, sign_languages: ["TİD"],
});
const sample = (id: string, sl = "TİD", extra: any = {}) => ({
  id, source: "eray_40hadis", source_name: "Eray Demir", sign_language: sl, label_source: "translation_match",
  youtube_url: `https://www.youtube.com/watch?v=${id}&t=0s`, video_id: id, start: 0, end: 17, read_start: 5.3,
  gloss: { [sl]: [{ text: "حسن", oov: false }, { text: "إسلام", oov: true }] }, isharati: true, ...extra,
});
const item = (ref: string, samples: any[]) => ({
  ref, collection: ref.split(":")[0], collection_name: "الأربعون النووية", number: ref.split(":")[1], also_in: [],
  text_ar: "نص الحديث", text_en: "The hadith text", text_tr: null, samples,
});

beforeEach(() => {
  vi.clearAllMocks();
  location.hash = "";
  (api.getHadithCoverage as any).mockResolvedValue(coverage);
  (api.getHadithList as any).mockImplementation(async (c: string) => [listItem(`${c}:12`)]);
  (api.getHadithItem as any).mockImplementation(async (ref: string) => item(ref, [sample("xVlmf7cW2LE")]));
  (api.getHadithPose as any).mockResolvedValue({ fps: 25, frames: [[[0, 0, 0]]] });
});
afterEach(cleanup);

describe("Signed Hadith section", () => {
  it("opens Nawawi's Forty first and its first hadith with text, signer and pose", async () => {
    render(<Hadith />);
    await waitFor(() => expect(api.getHadithList).toHaveBeenCalledWith("nawawi", ""));
    await waitFor(() => expect(api.getHadithItem).toHaveBeenCalledWith("nawawi:12"));
    await waitFor(() => expect(api.getHadithPose).toHaveBeenCalledWith("xVlmf7cW2LE"));
    expect(screen.getByText("نص الحديث")).toBeTruthy();
    expect(screen.getByText("The hadith text")).toBeTruthy();
    expect(screen.getAllByText("Eray Demir · TİD").length).toBe(2);   // the source card and the signer chip
  });

  it("shows the gloss with out-of-lexicon signs marked, and the original at its timestamp", async () => {
    render(<Hadith />);
    await waitFor(() => screen.getByText("إسلام"));
    expect(screen.getByText("إسلام").className).toContain("missing");
    expect(screen.getByText("حسن").className).not.toContain("missing");
    const link = screen.getByText(/hadith.watch/).closest("a")!;
    expect(link.getAttribute("href")).toBe("https://www.youtube.com/watch?v=xVlmf7cW2LE&t=0s");
    expect(document.querySelector("iframe")?.getAttribute("src")).toContain("/embed/xVlmf7cW2LE?start=0&end=17");
  });

  it("picking a collection lists its hadith; picking a signer loads that pose", async () => {
    (api.getHadithItem as any).mockImplementation(async (ref: string) =>
      item(ref, [sample("ArSL0000001", "ArSL"), sample("QXs2MiJT67k")]));
    render(<Hadith />);
    await waitFor(() => screen.getByText(/صحيح مسلم|Sahih Muslim/));
    fireEvent.click(screen.getByText("Sahih Muslim"));
    await waitFor(() => expect(api.getHadithList).toHaveBeenCalledWith("muslim", ""));
    await waitFor(() => expect(api.getHadithItem).toHaveBeenCalledWith("muslim:12"));
    await waitFor(() => screen.getByText("Eray Demir · TİD #2"));
    fireEvent.click(screen.getByText("Eray Demir · TİD #2"));
    await waitFor(() => expect(api.getHadithPose).toHaveBeenLastCalledWith("QXs2MiJT67k"));
  });

  it("the Academy's sign language filters the list and hides the filter", async () => {
    render(<Hadith signLanguage="TİD" />);
    await waitFor(() => expect(api.getHadithList).toHaveBeenCalledWith("nawawi", "TİD"));
    expect(screen.queryByText("hadith.allLangs")).toBeNull();
  });

  it("the sign-language filter reloads the list", async () => {
    render(<Hadith />);
    await waitFor(() => screen.getByText("ArSL"));
    fireEvent.click(screen.getByText("ArSL"));
    await waitFor(() => expect(api.getHadithList).toHaveBeenCalledWith("nawawi", "ArSL"));
  });

  it("an Academy link #/academy/hadith/<ref> opens that hadith", async () => {
    location.hash = "#/academy/hadith/muslim:2495";
    render(<Hadith />);
    await waitFor(() => expect(api.getHadithItem).toHaveBeenCalledWith("muslim:2495"));
    expect(api.getHadithList).toHaveBeenCalledWith("muslim", "");
    expect(api.getHadithItem).not.toHaveBeenCalledWith("muslim:12");
  });

  it("an empty dataset says so instead of failing", async () => {
    (api.getHadithList as any).mockResolvedValue([]);
    render(<Hadith />);
    await waitFor(() => screen.getByText("hadith.none"));
    expect(api.getHadithItem).not.toHaveBeenCalled();
  });
});

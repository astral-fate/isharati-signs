import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as api from "../api";
import * as i18n from "../i18n/i18n";
import * as state from "../state";
import { Quran } from "./Quran";

vi.mock("../api");
vi.mock("../i18n/i18n", () => ({ useI18n: () => ({ t: (k: string) => k }) }));
vi.mock("../state", () => ({
  useAppState: () => ({ avatar: "cartoon", outfit: "hijab" }),
}));
vi.mock("../components/AvatarView", () => ({ AvatarView: () => <canvas data-testid="avatar" /> }));
vi.mock("../components/KeypointCanvas", () => ({ KeypointCanvas: () => <canvas data-testid="kp" /> }));
vi.mock("../components/StyleSwitch", () => ({ StyleSwitch: () => null }));
vi.mock("../components/Counter", () => ({ Counter: ({ value }: any) => <span>{value}</span> }));
vi.mock("../player", () => ({ usePlayer: () => ({ t: 0, playing: false, toggle: vi.fn(), seek: vi.fn(), speed: 1, setSpeed: vi.fn() }) }));
vi.mock("framer-motion", () => ({
  motion: { div: ({ children, ...p }: any) => <div {...p}>{children}</div> },
  AnimatePresence: ({ children }: any) => children,
}));

const makeCoverage = () => ({
  sources: [{ key: "kfc", label: "KFC", segments: 10, recitation: 5, tafsir: 5, ayahs: 8, surahs: 2, hours: 1 }],
  surahs: Array.from({ length: 114 }, (_, i) => ({
    surah: i + 1, name: "",
    kfc: i === 0 ? { recitation: 1, tafsir: 0 } : { recitation: 0, tafsir: 0 },
    curriculum: { recitation: 0, tafsir: 0 },
    mukhtasar: { recitation: 0, tafsir: 0 },
  })),
});

const makeAyah = (segs: any[] = [], nearest: [number, number] | null = null) => ({
  surah: 1, ayah: 1, segments: segs, nearest,
});

beforeEach(() => {
  (api.getQuranCoverage as any).mockResolvedValue(makeCoverage());
  (api.getQuranAyah as any).mockResolvedValue(makeAyah());
  (api.getQuranPose as any).mockResolvedValue({ fps: 25, frames: [] });
});
afterEach(cleanup);

describe("Quran section", () => {
  it("renders 114 surah cells from coverage", async () => {
    render(<Quran />);
    await waitFor(() => expect(screen.getAllByRole("button", { name: /^\d+/ }).length).toBeGreaterThanOrEqual(114));
  });

  it("clicking a surah cell calls getQuranAyah with that surah and ayah 1", async () => {
    render(<Quran />);
    await waitFor(() => screen.getAllByRole("button", { name: /^1$/ }));
    const cells = screen.getAllByTitle(/^1/);
    if (cells[0]) fireEvent.click(cells[0]);
    await waitFor(() => expect(api.getQuranAyah).toHaveBeenCalledWith(1, 1));
  });

  it("shows nearest button when ayah has no segments", async () => {
    (api.getQuranAyah as any).mockResolvedValue(makeAyah([], [2, 255]));
    render(<Quran />);
    await waitFor(() => screen.getByText(/quran.goNearest/));
  });

  it("segment with isharati:false shows quran.original note", async () => {
    const seg = { id: "114_001", source: "kfc", type: "recitation", ayahs: [[114, 1]], seconds: 5, isharati: false };
    (api.getQuranAyah as any).mockResolvedValue(makeAyah([seg]));
    render(<Quran />);
    await waitFor(() => screen.getByText(`quran.source.kfc · quran.recitation`));
    fireEvent.click(screen.getByText(`quran.source.kfc · quran.recitation`));
    await waitFor(() => expect(screen.getByText("quran.original")).toBeTruthy());
  });

  it("segment without youtube_id renders no iframe", async () => {
    const seg = { id: "114_001", source: "kfc", type: "recitation", ayahs: [[114, 1]], seconds: 5, isharati: true };
    (api.getQuranAyah as any).mockResolvedValue(makeAyah([seg]));
    render(<Quran />);
    await waitFor(() => screen.getByText(`quran.source.kfc · quran.recitation`));
    fireEvent.click(screen.getByText(`quran.source.kfc · quran.recitation`));
    await waitFor(() => expect(api.getQuranPose).toHaveBeenCalled());
    expect(document.querySelector("iframe")).toBeNull();
  });
});

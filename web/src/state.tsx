import fallbackStats from "@static/stats.json";
import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { getAvatars, getClips, getStats } from "./api";
import { AVATAR_KEYS, OUTFITS, avatarForStyle, pickAvatar, styleOf, type AvatarStyle, type Outfit } from "./components/avatars";
import type { Clip, Stats } from "./types";

export type ThemeMode = "dark" | "light";

interface AppState {
  avatar: string; setAvatar: (m: string) => void; avatars: string[];
  style: AvatarStyle; setStyle: (s: AvatarStyle) => void;     // cartoon or realistic, switchable at any time
  outfit: Outfit; setOutfit: (o: Outfit) => void;             // the cartoon avatar's clothes
  theme: ThemeMode; toggleTheme: () => void;
  stats: Stats; clips: Clip[];
}
const Ctx = createContext<AppState | null>(null);
const KEY = "isharati.avatar", OUTFIT_KEY = "isharati.outfit", THEME_KEY = "isharati.theme";
const read = (k: string) => { try { return localStorage.getItem(k); } catch { return null; } };
const write = (k: string, v: string) => { try { localStorage.setItem(k, v); } catch { /* private mode */ } };

export function AppStateProvider({ children }: { children: ReactNode }) {
  const [avatars, setAvatars] = useState<string[]>(Object.keys(AVATAR_KEYS));
  const [avatar, setAvatarState] = useState(() => pickAvatar(read(KEY), Object.keys(AVATAR_KEYS)));
  const [last, setLast] = useState<Partial<Record<AvatarStyle, string>>>(() => ({ [styleOf(avatar)]: avatar }));
  const [outfit, setOutfitState] = useState<Outfit>(() => {
    const o = read(OUTFIT_KEY) as Outfit | null;
    return o && OUTFITS.includes(o) ? o : "hijab";
  });
  const [theme, setTheme] = useState<ThemeMode>(() => {
    const saved = read(THEME_KEY);
    if (saved === "light" || saved === "dark") return saved;
    return window.matchMedia?.("(prefers-color-scheme: light)").matches ? "light" : "dark";
  });

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    write(THEME_KEY, theme);
  }, [theme]);

  const toggleTheme = () => setTheme((t) => (t === "dark" ? "light" : "dark"));

  const [stats, setStats] = useState<Stats>(fallbackStats);  // the built-in figures until /api/stats answers
  const [clips, setClips] = useState<Clip[]>([]);
  useEffect(() => {
    getStats().then(setStats).catch(() => {});
    getClips().then(setClips).catch(() => {});
    getAvatars().then((list) => { setAvatars(list); setAvatarState((cur) => pickAvatar(cur, list)); }).catch(() => {});
  }, []);
  const setAvatar = (m: string) => {
    setAvatarState(m); write(KEY, m);
    setLast((l) => ({ ...l, [styleOf(m)]: m }));
  };
  const setStyle = (s: AvatarStyle) => setAvatar(avatarForStyle(s, last, avatars));
  const setOutfit = (o: Outfit) => { setOutfitState(o); write(OUTFIT_KEY, o); };
  return (
    <Ctx.Provider value={{ avatar, setAvatar, avatars, style: styleOf(avatar), setStyle, outfit, setOutfit, theme, toggleTheme, stats, clips }}>
      {children}
    </Ctx.Provider>
  );
}

export function useAppState(): AppState {
  const v = useContext(Ctx);
  if (!v) throw new Error("useAppState outside AppStateProvider");
  return v;
}

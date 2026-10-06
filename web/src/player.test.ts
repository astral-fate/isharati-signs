import { act, renderHook } from "@testing-library/react";
import { usePlayer } from "./player";

let queue: Map<number, FrameRequestCallback>, nextId: number, clock: number;
beforeEach(() => {
  queue = new Map(); nextId = 1; clock = 0;
  vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => { queue.set(nextId, cb); return nextId++; });
  vi.stubGlobal("cancelAnimationFrame", (id: number) => { queue.delete(id); });
  vi.spyOn(performance, "now").mockImplementation(() => clock);
});
afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks(); });
const tick = (ms: number) => act(() => {
  clock += ms;
  const cbs = [...queue.values()]; queue.clear();
  cbs.forEach((cb) => cb(clock));
});

test("autoplay advances t by elapsed x speed", () => {
  const { result } = renderHook(() => usePlayer(10, { autoplay: true }));
  act(() => result.current.setSpeed(1));
  tick(100);
  expect(result.current.t).toBeCloseTo(0.1);
});

test("non-loop stops exactly at duration", () => {
  const { result } = renderHook(() => usePlayer(0.15, { autoplay: true }));
  act(() => result.current.setSpeed(1));
  tick(100); tick(100);
  expect(result.current.t).toBe(0.15);
  expect(result.current.playing).toBe(false);
});

test("loop wraps past the end", () => {
  const { result } = renderHook(() => usePlayer(0.15, { autoplay: true, loop: true }));
  act(() => result.current.setSpeed(1));
  tick(100); tick(100);
  expect(result.current.t).toBeCloseTo(0.05);
  expect(result.current.playing).toBe(true);
});

test("seek while playing continues from the seek point", () => {
  const { result } = renderHook(() => usePlayer(10, { autoplay: true }));
  act(() => result.current.setSpeed(1));
  tick(100);
  act(() => result.current.seek(5));
  tick(100);
  expect(result.current.t).toBeCloseTo(5.1);
});

test("play after the end restarts from 0", () => {
  const { result } = renderHook(() => usePlayer(0.1, { autoplay: true }));
  act(() => result.current.setSpeed(1));
  tick(100);
  expect(result.current.playing).toBe(false);
  act(() => result.current.play());
  expect(result.current.t).toBe(0);
  expect(result.current.playing).toBe(true);
});

test("seek(NaN) is ignored", () => {
  const { result } = renderHook(() => usePlayer(10));
  act(() => result.current.seek(3));
  act(() => result.current.seek(NaN));
  expect(result.current.t).toBe(3);
});

test("a long frame gap is clamped to 0.1 s", () => {
  const { result } = renderHook(() => usePlayer(10, { autoplay: true }));
  act(() => result.current.setSpeed(1));
  tick(5000);
  expect(result.current.t).toBeCloseTo(0.1);
});

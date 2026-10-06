import { afterEach, beforeAll, beforeEach, expect, test, vi, type Mock } from "vitest";
import { createElement } from "react";
import { act, render as mount } from "@testing-library/react";

interface Fake {
  load: Mock; render: Mock; setFrames: Mock;
  resize: Mock; renderer: { dispose: Mock };
  resolve: () => void; reject: (e: Error) => void;
}
const made: Fake[] = [];
(globalThis as unknown as { __made: Fake[] }).__made = made;  // the hoisted mock factory pushes here

vi.mock("@static/avatar.js", () => {
  return {
    SignAvatar: class {
      renderer = { dispose: vi.fn() }; render = vi.fn(); setFrames = vi.fn(); resize = vi.fn();
      resolve!: () => void; reject!: (e: Error) => void;
      load = vi.fn(() => new Promise((res, rej) => { this.resolve = () => res(this); this.reject = rej; }));
      constructor() { (globalThis as unknown as { __made: Fake[] }).__made.push(this as unknown as Fake); }
    },
  };
});
vi.mock("@static/outfits.js", () => ({ applyOutfit: () => {} }));

beforeEach(() => {
  made.length = 0;
  vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} });
  vi.spyOn(console, "error").mockImplementation(() => {});
});
afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks(); });

let AvatarView: typeof import("./AvatarView").AvatarView;
beforeAll(async () => { ({ AvatarView } = await import("./AvatarView")); });

const view = (model: string, onError = () => {}, time = 0) =>
  createElement(AvatarView, { model, frames: null, time, onError });

test("changing model disposes the previous instance and never renders it afterwards", async () => {
  const { rerender } = mount(view("a.vrm"));
  await act(async () => { made[0].resolve(); });
  expect(made[0].render).toHaveBeenCalled();
  const before = made[0].render.mock.calls.length;
  rerender(view("b.vrm", () => {}, 1));
  expect(made[0].renderer.dispose).toHaveBeenCalledTimes(1);
  await act(async () => { made[1].resolve(); });
  rerender(view("b.vrm", () => {}, 2));
  expect(made[0].render.mock.calls.length).toBe(before);
  expect(made[1].render).toHaveBeenCalled();
});

test("a rejected load disposes the instance and calls onError", async () => {
  const onError = vi.fn();
  mount(view("a.vrm", onError));
  await act(async () => { made[0].reject(new Error("boom")); });
  expect(made[0].renderer.dispose).toHaveBeenCalledTimes(1);
  expect(onError).toHaveBeenCalledTimes(1);
});

test("unmounting disposes", async () => {
  const { unmount } = mount(view("a.vrm"));
  await act(async () => { made[0].resolve(); });
  unmount();
  expect(made[0].renderer.dispose).toHaveBeenCalledTimes(1);
});

test("a load that resolves after a model change is disposed and never rendered", async () => {
  const onError = vi.fn();
  const { rerender } = mount(view("a.vrm", onError));
  rerender(view("b.vrm", onError));
  expect(made[0].renderer.dispose).toHaveBeenCalledTimes(1);
  await act(async () => { made[0].resolve(); });
  expect(made[0].render).not.toHaveBeenCalled();
  expect(made[0].renderer.dispose).toHaveBeenCalledTimes(1);
  expect(onError).not.toHaveBeenCalled();
});

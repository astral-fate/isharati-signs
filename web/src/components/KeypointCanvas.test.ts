import { poseBounds } from "./KeypointCanvas";

const frame = (x: number, y: number) => Array.from({ length: 50 }, () => [x, y, 0]);

test("bounds span every frame and ignore non-finite values", () => {
  const f1 = frame(-1, -2), f2 = frame(1, 3);
  f2[10] = [NaN, Infinity, 0];
  expect(poseBounds([f1, f2])).toEqual({ minX: -1, maxX: 1, minY: -2, maxY: 3 });
});

test("a single still frame still has a usable box", () => {
  const b = poseBounds([frame(0, 0)]);
  expect(b.maxX - b.minX).toBeGreaterThan(0);
  expect(b.maxY - b.minY).toBeGreaterThan(0);
});

import { render, screen } from "@testing-library/react";
import type { Segment } from "../types";
import { GlossChips } from "./GlossChips";

const segs: Segment[] = [
  { label: "ALLAH", kind: "sign", start_s: 0.2, end_s: 1.0 },
  { label: "be", kind: "missing", start_s: 1.3, end_s: 1.7 },
  { label: "PRAYER", kind: "sign", start_s: 2.0, end_s: 3.0 },
];

test("the chip being signed now is highlighted; missing signs are red", () => {
  render(<GlossChips segments={segs} time={2.5} onSeek={() => {}} />);
  expect(screen.getByText("PRAYER")).toHaveClass("now");
  expect(screen.getByText("ALLAH")).not.toHaveClass("now");
  expect(screen.getByText("be")).toHaveClass("missing");
});

test("clicking a chip seeks to its start", () => {
  const onSeek = vi.fn();
  render(<GlossChips segments={segs} time={0} onSeek={onSeek} />);
  screen.getByText("PRAYER").click();
  expect(onSeek).toHaveBeenCalledWith(2.0);
});

test("no segments renders an empty list", () => {
  const { container } = render(<GlossChips segments={[]} time={0} onSeek={() => {}} />);
  expect(container.querySelectorAll(".chip")).toHaveLength(0);
});

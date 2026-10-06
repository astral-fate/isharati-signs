import { render, screen } from "@testing-library/react";
import { I18nProvider } from "../i18n/i18n";
import { Counter } from "./Counter";

test("shows the final value with grouping when not animated", () => {
  render(<I18nProvider initial="en"><Counter value={3213} duration={0} /></I18nProvider>);
  expect(screen.getByText("3,213")).toBeInTheDocument();
});

test("formats a share as a whole percentage", () => {
  render(<I18nProvider initial="en"><Counter value={0.6316} format="pct" duration={0} /></I18nProvider>);
  expect(screen.getByText("63%")).toBeInTheDocument();
});

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { I18nProvider } from "../i18n/i18n";
import { Pricing } from "./Pricing";
import { UseCases } from "./UseCases";

describe("Commercial sections: UseCases and Pricing", () => {
  it("renders all 4 use cases with titles and tags", () => {
    render(
      <I18nProvider initial="en">
        <UseCases />
      </I18nProvider>
    );

    expect(screen.getByText(/Mosques & Juma'a Khutbahs/i)).toBeDefined();
    expect(screen.getByText(/Islamic Schools & Madrasahs/i)).toBeDefined();
    expect(screen.getByText(/Dawah & Media Outreach/i)).toBeDefined();
    expect(screen.getByText(/Deaf Ummah Charities/i)).toBeDefined();
  });

  it("renders 4 pricing tiers with billing toggle", () => {
    render(
      <I18nProvider initial="en">
        <Pricing />
      </I18nProvider>
    );

    // Default monthly pricing
    expect(screen.getByText("$29")).toBeDefined();
    expect(screen.getByText("$199")).toBeDefined();

    // Toggle to yearly
    const toggle = screen.getByRole("switch");
    fireEvent.click(toggle);

    // Discounted annual prices
    expect(screen.getByText("$24")).toBeDefined();
    expect(screen.getByText("$169")).toBeDefined();
  });

  it("displays unit economics cost efficiency comparison", () => {
    render(
      <I18nProvider initial="en">
        <Pricing />
      </I18nProvider>
    );

    expect(screen.getByText(/\$0\.0025/)).toBeDefined();
  });
});

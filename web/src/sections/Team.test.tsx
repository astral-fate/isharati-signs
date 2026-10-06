import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { I18nProvider } from "../i18n/i18n";
import { Team } from "./Team";

function renderWithI18n(ui: React.ReactElement) {
  return render(<I18nProvider initial="en">{ui}</I18nProvider>);
}

describe("Team section", () => {
  it("renders both leadership profiles with portrait frames and QR links", () => {
    renderWithI18n(<Team />);
    expect(screen.getByText(/Fatimah Emad Eldin/i)).toBeInTheDocument();
    expect(screen.getByText(/Dr\. Asmaa Al-Mirghani/i)).toBeInTheDocument();

    const images = screen.getAllByRole("img");
    const photoSrcs = images.map((img) => img.getAttribute("src"));
    expect(photoSrcs).toContain("/team/fatimah.png");
    expect(photoSrcs).toContain("/team/asmaa.jpg");
    expect(photoSrcs).toContain("/team/qr_linkedin.png");
    expect(photoSrcs).toContain("/team/qr_youtube.png");
  });

  it("renders credentials, bullets, and external social links", () => {
    renderWithI18n(<Team />);
    expect(screen.getAllByText(/ARSL-GEN/i).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText(/ICCV 2025/i).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText(/Applied Statistics/i).length).toBeGreaterThanOrEqual(1);

    const links = screen.getAllByRole("link");
    const hrefs = links.map((a) => a.getAttribute("href"));
    expect(hrefs).toContain("https://www.linkedin.com/in/astral-fate");
    expect(hrefs).toContain("https://www.youtube.com/channel/UC0AyY5dK_0iL31BMiphTh3A");
  });
});

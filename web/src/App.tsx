import { useEffect, useState } from "react";
import { Academy } from "./sections/Academy";
import { Avatars } from "./sections/Avatars";
import { Footer } from "./sections/Footer";
import { AcademyPromo, Features } from "./sections/Features";
import { Hadith } from "./sections/Hadith";
import { sacredPage } from "./sections/hadithData";
import { Quran } from "./sections/Quran";
import { SacredPreview } from "./sections/SacredPreview";
import { Hero } from "./sections/Hero";
import { HowItWorks } from "./sections/HowItWorks";
import { Lexicon } from "./sections/Lexicon";
import { Nav } from "./sections/Nav";
import { Pricing } from "./sections/Pricing";
import { ReviewStudio } from "./sections/ReviewStudio";
import { Team } from "./sections/Team";
import { Translator } from "./sections/Translator";
import { UseCases } from "./sections/UseCases";
import { AppStateProvider } from "./state";

function isReviewRoute() {
  return (
    typeof window !== "undefined" &&
    (window.location.pathname === "/review" ||
      window.location.hash === "#/review" ||
      window.location.hash.startsWith("#/review"))
  );
}

function isAcademyRoute() {
  return typeof window !== "undefined" && (window.location.pathname === "/academy" || window.location.hash.startsWith("#/academy"));
}

// the Sign Mushaf (#/quran) and signed hadith (#/hadith[/ref]) pages; #/academy/quran and #/academy/hadith are aliases
const sacredRoute = () => (typeof window !== "undefined" ? sacredPage(window.location.hash) : null);

export default function App() {
  const [isReview, setIsReview] = useState(isReviewRoute);
  const [isAcademy, setIsAcademy] = useState(isAcademyRoute);
  const [sacred, setSacred] = useState(sacredRoute);

  useEffect(() => {
    const handleRoute = () => { setIsReview(isReviewRoute()); setIsAcademy(isAcademyRoute()); setSacred(sacredRoute()); };
    window.addEventListener("hashchange", handleRoute);
    window.addEventListener("popstate", handleRoute);
    return () => {
      window.removeEventListener("hashchange", handleRoute);
      window.removeEventListener("popstate", handleRoute);
    };
  }, []);

  return (
    <AppStateProvider>
      <Nav />
      {sacred ? (
        <main style={{ paddingTop: 80, minHeight: "80vh" }}>
          {sacred === "quran" ? <Quran /> : <Hadith />}
        </main>
      ) : isAcademy ? (
        <main style={{ paddingTop: 80, minHeight: "80vh" }}>
          <Academy />
        </main>
      ) : isReview ? (
        <main style={{ paddingTop: 80, minHeight: "80vh" }}>
          <ReviewStudio />
        </main>
      ) : (
        <>
          <Hero />
          <main>
            <Features />
            <Translator />
            <HowItWorks />
            <Lexicon />
            <SacredPreview />
            <AcademyPromo />
            <Avatars />
            <UseCases />
            <Pricing />
            <Team />
          </main>
        </>
      )}
      <Footer />
    </AppStateProvider>
  );
}


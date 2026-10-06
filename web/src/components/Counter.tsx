import { animate, useInView } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { useI18n } from "../i18n/i18n";

const LOCALE = { en: "en", ar: "ar-u-nu-latn", tr: "tr", ur: "ur-u-nu-latn" } as const;

export function Counter({ value, format = "int", duration = 1.6 }: { value: number; format?: "int" | "pct" | "fixed1"; duration?: number }) {
  const { ui } = useI18n();
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true });
  const [shown, setShown] = useState(duration === 0 ? value : 0);
  useEffect(() => {
    if (duration === 0) { setShown(value); return; }
    if (!inView) return;
    const c = animate(0, value, { duration, ease: "easeOut", onUpdate: setShown });
    return () => c.stop();
  }, [inView, value, duration]);
  const nf = new Intl.NumberFormat(LOCALE[ui], format === "pct" ? { style: "percent", maximumFractionDigits: 0 }
                                              : format === "fixed1" ? { minimumFractionDigits: 1, maximumFractionDigits: 1 }
                                              : { maximumFractionDigits: 0 });
  return <span ref={ref} className="counter">{nf.format(format === "pct" ? shown : format === "fixed1" ? shown : Math.round(shown))}</span>;
}

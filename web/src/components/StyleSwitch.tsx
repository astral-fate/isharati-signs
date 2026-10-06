import { useI18n } from "../i18n/i18n";
import { useAppState } from "../state";
import { AvatarPicker } from "./AvatarPicker";
import { OUTFITS, STYLES } from "./avatars";

export function StyleSwitch() {
  const { t } = useI18n();
  const { avatar, setAvatar, style, setStyle, outfit, setOutfit } = useAppState();
  return (
    <div className="style-switch">
      <div className="seg" role="radiogroup" aria-label={t("style.label")}>
        {STYLES.map((s) => (
          <button key={s} role="radio" aria-checked={s === style} className={s === style ? "on" : ""}
                  onClick={() => setStyle(s)}>{t(`style.${s}`)}</button>
        ))}
      </div>
      <p className="note">{t(`style.${style}Note`)}</p>
      {style === "realistic"
        ? <AvatarPicker value={avatar} onChange={setAvatar} />
        : (
          <div className="seg" role="radiogroup" aria-label={t("outfit.label")}>
            {OUTFITS.map((o) => (
              <button key={o} role="radio" aria-checked={o === outfit} className={o === outfit ? "on" : ""}
                      onClick={() => setOutfit(o)}>{t(`outfit.${o}`)}</button>
            ))}
          </div>
        )}
    </div>
  );
}

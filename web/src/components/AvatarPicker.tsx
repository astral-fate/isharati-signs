import { useI18n } from "../i18n/i18n";
import { useAppState } from "../state";
import { AVATAR_KEYS, avatarsFor, styleOf } from "./avatars";

/** The avatars of one style, as thumbnails (each avatar's clip poster once clips exist). */
export function AvatarPicker({ value, onChange }: { value: string; onChange: (m: string) => void }) {
  const { t } = useI18n();
  const { avatars, clips } = useAppState();
  return (
    <div className="pickers" role="radiogroup" aria-label={t("try.pick")}>
      {avatarsFor(styleOf(value), avatars).map((m) => {
        const poster = clips.find((c) => c.avatar === m)?.poster;
        const name = t(AVATAR_KEYS[m] ?? m);
        return (
          <button key={m} type="button" role="radio" aria-checked={m === value} title={name}
                  className={"picker" + (m === value ? " on" : "")} onClick={() => onChange(m)}
                  style={poster ? { backgroundImage: `url(${poster})` } : undefined}>
            {poster ? "" : name}
          </button>
        );
      })}
    </div>
  );
}

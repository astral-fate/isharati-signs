import { useI18n } from "../i18n/i18n";

export function Footer() {
  const { t } = useI18n();
  return <footer><div className="wrap">{t("foot.note")} · <a href="/classic">{t("foot.classic")}</a></div></footer>;
}

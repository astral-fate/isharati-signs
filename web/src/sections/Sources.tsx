import { useI18n } from "../i18n/i18n";
import { useAppState } from "../state";

export function Sources() {
  const { t } = useI18n();
  const { stats } = useAppState();
  return (
    <section id="sources">
      <div className="wrap">
        <h2>{t("src.title")}</h2>
        <p className="lead">{t("src.lead")}</p>
        <div className="panel" style={{ overflowX: "auto" }}>
          <table>
            <thead><tr><th>{t("src.source")}</th><th>{t("src.licence")}</th><th>{t("src.use")}</th></tr></thead>
            <tbody dir="ltr">{stats.licences.map((l) => <tr key={l.source}><td>{l.source}</td><td>{l.licence}</td><td>{l.use}</td></tr>)}</tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

import type { Stats } from "../types";

type Props = { stats: Stats | null };

export function Metrics({ stats }: Props) {
  // Use most recent day from daily arrays (today)
  const todayIn =
  stats?.daily.in[stats.daily.in.length - 1] ?? null;

  const todayOut =
  stats?.daily.out[stats.daily.out.length - 1] ?? null;
  const net =
    todayIn !== null && todayOut !== null ? todayIn - todayOut : null;

  return (
    <section className="metrics" aria-label="Today's totals">
      <article className="card">
        <h2 className="card__label">Today in</h2>
        <p className="card__value card__value--in">
          {todayIn === null ? "—" : todayIn}
        </p>
      </article>

      <article className="card">
        <h2 className="card__label">Today out</h2>
        <p className="card__value card__value--out">
          {todayOut === null ? "—" : todayOut}
        </p>
      </article>

      <article className="card">
        <h2 className="card__label">Net inside</h2>
        <p
          className="card__value"
          style={{
            color:
              net === null
                ? "var(--text)"
                : net > 0
                ? "var(--accent-in)"
                : net < 0
                ? "var(--accent-out)"
                : "var(--text)",
          }}
        >
          {net === null ? "—" : net}
        </p>
        <p className="card__hint">
          Today's net change (in − out)
        </p>
      </article>
    </section>
  );
}
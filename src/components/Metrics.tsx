import type { Stats } from "../types";

type Props = { stats: Stats | null };

export function Metrics({ stats }: Props) {
  const tin = stats?.totals.in ?? null;
  const tout = stats?.totals.out ?? null;
  const net = tin !== null && tout !== null ? tin - tout : null;

  return (
    <section className="metrics" aria-label="Totals">
      <article className="card">
        <h2 className="card__label">Overall in</h2>
        <p className="card__value card__value--in">{tin === null ? "—" : tin}</p>
      </article>
      <article className="card">
        <h2 className="card__label">Overall out</h2>
        <p className="card__value card__value--out">{tout === null ? "—" : tout}</p>
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
        <p className="card__hint">In minus out (from recorded events)</p>
      </article>
    </section>
  );
}

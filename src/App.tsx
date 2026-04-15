import { useCallback, useEffect, useState } from "react";
import { fetchStats } from "./api";
import type { Stats } from "./types";
import { Clock } from "./components/Clock";
import { Metrics } from "./components/Metrics";
import { DashboardCharts } from "./components/DashboardCharts";
import { ThemeToggle } from "./components/ThemeToggle";
import { useTheme } from "./useTheme";
import { getInitialHourly24h, persistHourly24h } from "./hourlyFormat";

const POLL_MS = 5000;

type DashboardPage = "overview" | "occupancy" | "trends" | "patterns";

const PAGE_TITLES: Record<DashboardPage, string> = {
  overview: "Overview",
  occupancy: "Occupancy",
  trends: "Trends",
  patterns: "Traffic patterns",
};

const PAGE_SUBTITLES: Record<DashboardPage, string> = {
  overview: "Live summary of people flow and pacing",
  occupancy: "Inside vs typical day and hourly net flow",
  trends: "Hourly, daily, and weekly movement over time",
  patterns: "Most active hours and directional flow by hour",
};

export default function App() {
  const { theme, toggleTheme } = useTheme();
  const [stats, setStats] = useState<Stats | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [hourly24h, setHourly24h] = useState(getInitialHourly24h);
  const [page, setPage] = useState<DashboardPage>("overview");

  useEffect(() => {
    persistHourly24h(hourly24h);
  }, [hourly24h]);

  const load = useCallback(async () => {
    try {
      const data = await fetchStats();
      setStats(data);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load");
    }
  }, []);

  useEffect(() => {
    load();
    const id = window.setInterval(load, POLL_MS);
    return () => window.clearInterval(id);
  }, [load]);

  const statusText =
    error ??
    (stats
      ? `Last updated ${new Date(stats.updated_at).toLocaleTimeString()} (server UTC)`
      : "Loading…");

  return (
    <div className="dashboard-shell">
      <aside className="dashboard-sidebar">
        <div className="dashboard-sidebar__brand">
          <div className="dashboard-sidebar__eyebrow">Cal Poly Pomona</div>
          <h1 className="dashboard-sidebar__title">Maker Space Dashboard</h1>
          <p className="dashboard-sidebar__subtitle">Daily usage, flow, and activity insights</p>
        </div>

        <nav className="dashboard-sidebar__nav" aria-label="Dashboard navigation">
          <button
            type="button"
            className={`dashboard-sidebar__link${
              page === "overview" ? " dashboard-sidebar__link--active" : ""
            }`}
            onClick={() => setPage("overview")}
          >
            Overview
          </button>

          <button
            type="button"
            className={`dashboard-sidebar__link${
              page === "occupancy" ? " dashboard-sidebar__link--active" : ""
            }`}
            onClick={() => setPage("occupancy")}
          >
            Occupancy
          </button>

          <button
            type="button"
            className={`dashboard-sidebar__link${
              page === "trends" ? " dashboard-sidebar__link--active" : ""
            }`}
            onClick={() => setPage("trends")}
          >
            Trends
          </button>

          <button
            type="button"
            className={`dashboard-sidebar__link${
              page === "patterns" ? " dashboard-sidebar__link--active" : ""
            }`}
            onClick={() => setPage("patterns")}
          >
            Traffic patterns
          </button>
        </nav>
      </aside>

      <main className="dashboard-main">
        <header className="dashboard-topbar">
          <div>
            <div className="dashboard-topbar__title">{PAGE_TITLES[page]}</div>
            <div className="dashboard-topbar__subtitle">{PAGE_SUBTITLES[page]}</div>
          </div>

          <div className="dashboard-topbar__actions">
            <ThemeToggle theme={theme} onToggle={toggleTheme} />
            <Clock />
          </div>
        </header>

        {page === "overview" && (
          <section className="dashboard-section">
            <Metrics stats={stats} />
          </section>
        )}

        <section className="dashboard-section">
          <p className={`status${error ? " status--error" : ""}`} role="status">
            {statusText}
          </p>
        </section>

        <section className="dashboard-section">
          <DashboardCharts
            stats={stats}
            theme={theme}
            hourly24h={hourly24h}
            onHourly24hChange={setHourly24h}
            page={page}
          />
        </section>
      </main>
    </div>
  );
}
/* VERCEL TESTING */

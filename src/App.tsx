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

export default function App() {
  const { theme, toggleTheme } = useTheme();
  const [stats, setStats] = useState<Stats | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [hourly24h, setHourly24h] = useState(getInitialHourly24h);

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
    <div className="page">
      <header className="header">
        <div>
          <span className="header__title">People count</span>
          <span className="header__subtitle">Ingress & egress</span>
        </div>
        <div className="header__actions">
          <ThemeToggle theme={theme} onToggle={toggleTheme} />
          <Clock />
        </div>
      </header>

      <Metrics stats={stats} />

      <p className={`status${error ? " status--error" : ""}`} role="status">
        {statusText}
      </p>

      <DashboardCharts
        stats={stats}
        theme={theme}
        hourly24h={hourly24h}
        onHourly24hChange={setHourly24h}
      />
    </div>
  );
}

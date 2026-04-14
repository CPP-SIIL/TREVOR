import { useMemo } from "react";
import { Line, Bar } from "react-chartjs-2";
import type { ChartOptions } from "chart.js";
import type { Stats } from "../types";
import { getChartPalette, type ThemeMode } from "../chartPalette";
import { formatHourlyAxisLabel } from "../hourlyFormat";

function buildLineOptions(
  tick: string,
  grid: string,
  beginAtZero = true
): ChartOptions<"line"> {
  const legendLabels = {
    color: tick,
    font: { family: "'DM Sans', sans-serif", size: 12 },
  };

  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    plugins: { legend: { labels: legendLabels } },
    scales: {
      x: {
        grid: { color: grid },
        ticks: { color: tick, maxRotation: 45, minRotation: 0 },
      },
      y: {
        beginAtZero,
        grid: { color: grid },
        ticks: { color: tick, precision: 0 },
      },
    },
  };
}

function buildBarOptions(
  tick: string,
  grid: string,
  beginAtZero = true
): ChartOptions<"bar"> {
  const legendLabels = {
    color: tick,
    font: { family: "'DM Sans', sans-serif", size: 12 },
  };

  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    plugins: { legend: { labels: legendLabels } },
    scales: {
      x: {
        grid: { color: grid },
        ticks: { color: tick, maxRotation: 45, minRotation: 0 },
      },
      y: {
        beginAtZero,
        grid: { color: grid },
        ticks: { color: tick, precision: 0 },
      },
    },
  };
}

function buildBarStackedOptions(
  base: ChartOptions<"bar">
): ChartOptions<"bar"> {
  return {
    ...base,
    scales: {
      x: { ...base.scales?.x, stacked: true },
      y: { ...base.scales?.y, stacked: true },
    },
  };
}

type DashboardPage = "overview" | "occupancy" | "trends" | "patterns";

type Props = {
  stats: Stats | null;
  theme: ThemeMode;
  hourly24h: boolean;
  onHourly24hChange: (value: boolean) => void;
  page: DashboardPage;
};

type ChartCardProps = {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
  action?: React.ReactNode;
};

function ChartCard({ title, subtitle, children, action }: ChartCardProps) {
  return (
    <article className="card card--chart">
      <div className="card__heading-row">
        <h2 className="card__heading card__heading--inline">{title}</h2>
        {action}
      </div>
      {subtitle ? <p className="card__sub">{subtitle}</p> : null}
      <div className="chart-wrap">{children}</div>
    </article>
  );
}

export function DashboardCharts({
  stats,
  theme,
  hourly24h,
  onHourly24hChange,
  page,
}: Props) {
  const palette = useMemo(() => getChartPalette(theme), [theme]);

  const lineOptions = useMemo(
    () => buildLineOptions(palette.tick, palette.grid, true),
    [palette.tick, palette.grid]
  );

  const lineOptionsOccupancy = useMemo(
    () => buildLineOptions(palette.tick, palette.grid, false),
    [palette.tick, palette.grid]
  );

  const barOptions = useMemo(
    () => buildBarOptions(palette.tick, palette.grid, true),
    [palette.tick, palette.grid]
  );

  const barOptionsNet = useMemo(
    () => buildBarOptions(palette.tick, palette.grid, false),
    [palette.tick, palette.grid]
  );

  const barStackedOptions = useMemo(
    () => buildBarStackedOptions(barOptions),
    [barOptions]
  );

  const hourlyLabels = useMemo(() => {
    if (!stats) return [];
    return stats.hourly.labels.map((label) =>
      formatHourlyAxisLabel(label, hourly24h)
    );
  }, [stats, hourly24h]);

  const occupancyAxisLabels = useMemo(() => {
    if (!stats) return [];
    return stats.occupancy.labels_24h.map((label) =>
      formatHourlyAxisLabel(label, hourly24h)
    );
  }, [stats, hourly24h]);

  const hourly = useMemo(() => {
    if (!stats) return null;
    return {
      labels: hourlyLabels,
      datasets: [
        {
          label: "In",
          data: stats.hourly.in,
          borderColor: palette.in,
          backgroundColor: palette.inFill,
          fill: true,
          tension: 0.25,
        },
        {
          label: "Out",
          data: stats.hourly.out,
          borderColor: palette.out,
          backgroundColor: palette.outFill,
          fill: true,
          tension: 0.25,
        },
      ],
    };
  }, [stats, hourlyLabels, palette]);

  const daily = useMemo(() => {
    if (!stats) return null;
    return {
      labels: stats.daily.labels,
      datasets: [
        {
          label: "In",
          data: stats.daily.in,
          borderColor: palette.in,
          backgroundColor: palette.inFill,
          fill: true,
          tension: 0.2,
        },
        {
          label: "Out",
          data: stats.daily.out,
          borderColor: palette.out,
          backgroundColor: palette.outFill,
          fill: true,
          tension: 0.2,
        },
      ],
    };
  }, [stats, palette]);

  const weekly = useMemo(() => {
    if (!stats) return null;
    return {
      labels: stats.weekly.labels,
      datasets: [
        { label: "In", data: stats.weekly.in, backgroundColor: palette.in },
        { label: "Out", data: stats.weekly.out, backgroundColor: palette.out },
      ],
    };
  }, [stats, palette]);

  const popularTotal = useMemo(() => {
    if (!stats) return null;
    return {
      labels: stats.popular_hours.labels,
      datasets: [
        {
          label: "In + out (net)",
          data: stats.popular_hours.total,
          backgroundColor: palette.total,
        },
      ],
    };
  }, [stats, palette]);

  const popularSplit = useMemo(() => {
    if (!stats) return null;
    return {
      labels: stats.popular_hours.labels,
      datasets: [
        {
          label: "In (net)",
          data: stats.popular_hours.in,
          backgroundColor: palette.in,
        },
        {
          label: "Out (net)",
          data: stats.popular_hours.out,
          backgroundColor: palette.out,
        },
      ],
    };
  }, [stats, palette]);

  const occupancyCumulative = useMemo(() => {
    if (!stats) return null;

    const occ = stats.occupancy;
    const baselineLabel =
      occ.baseline_sample_days > 0
        ? `Typical ${occ.weekday_label} (avg of ${occ.baseline_sample_days} prior days)`
        : `Typical ${occ.weekday_label} (no baseline yet)`;

    return {
      labels: occupancyAxisLabels,
      datasets: [
        {
          label: "Net inside today (so far)",
          data: occ.today_cumulative,
          borderColor: palette.occupancyLine,
          backgroundColor: "transparent",
          tension: 0.25,
          spanGaps: false,
        },
        {
          label: baselineLabel,
          data: occ.baseline_cumulative,
          borderColor: palette.baseline,
          backgroundColor: "transparent",
          borderDash: [6, 4],
          tension: 0.2,
          pointRadius: 0,
        },
      ],
    };
  }, [stats, occupancyAxisLabels, palette]);

  const occupancyHourlyCompare = useMemo(() => {
    if (!stats) return null;

    const occ = stats.occupancy;
    return {
      labels: occupancyAxisLabels,
      datasets: [
        {
          label: "This hour today",
          data: occ.today_hourly_net,
          backgroundColor: palette.occupancyLine,
        },
        {
          label: "Avg this weekday",
          data: occ.baseline_hourly_net_avg,
          backgroundColor: palette.baseline,
        },
      ],
    };
  }, [stats, occupancyAxisLabels, palette]);

  if (!stats) {
    return (
      <section className="charts" aria-label="Charts">
        <article className="card card--chart">
          <p className="card__sub" style={{ margin: 0 }}>
            Load stats to see charts.
          </p>
        </article>
      </section>
    );
  }

  const occ = stats.occupancy;

  return (
    <section className="charts" aria-label="Charts">
      {page === "overview" && (
        <>
          <ChartCard
            title="Net inside vs typical day"
            subtitle={`Cumulative net people inside since midnight ${occ.timezone_note}. Use for pacing, not absolute capacity.`}
          >
            {occupancyCumulative && (
              <Line data={occupancyCumulative} options={lineOptionsOccupancy} />
            )}
          </ChartCard>

          <ChartCard
            title="Hourly net flow vs baseline"
            subtitle='Per-hour net change for today so far vs the average for this weekday.'
          >
            {occupancyHourlyCompare && (
              <Bar data={occupancyHourlyCompare} options={barOptionsNet} />
            )}
          </ChartCard>
        </>
      )}

      {page === "occupancy" && (
        <>
          <ChartCard
            title="Net inside vs typical day"
            subtitle={`Cumulative net people inside since midnight ${occ.timezone_note}: each step adds (net entries − net exits) for that hour. The dashed line is the average curve for past ${occ.weekday_label}s. Starts from zero at midnight.`}
          >
            {occupancyCumulative && (
              <Line data={occupancyCumulative} options={lineOptionsOccupancy} />
            )}
          </ChartCard>

          <ChartCard
            title="Hourly net flow vs baseline"
            subtitle='Per-hour net change (in − out) for today so far vs the average for this weekday at the same hour.'
          >
            {occupancyHourlyCompare && (
              <Bar data={occupancyHourlyCompare} options={barOptionsNet} />
            )}
          </ChartCard>
        </>
      )}

      {page === "trends" && (
        <>
          <ChartCard
            title="Hourly (last 24 h, UTC buckets)"
            subtitle={`Axis uses ${hourly24h ? "24-hour" : "12-hour"} labels (UTC).`}
            action={
              <button
                type="button"
                className="toggle-btn toggle-btn--small"
                onClick={() => onHourly24hChange(!hourly24h)}
                aria-pressed={hourly24h}
              >
                {hourly24h ? "12-hour clock" : "24-hour (military)"}
              </button>
            }
          >
            {hourly && <Line data={hourly} options={lineOptions} />}
          </ChartCard>

          <ChartCard title="Daily (last 14 days, UTC)">
            {daily && <Line data={daily} options={lineOptions} />}
          </ChartCard>

          <ChartCard
            title="Weekly (last 8 weeks, Mon UTC)"
            subtitle="Each bar is labeled with the week's date span and ISO week number."
          >
            {weekly && <Bar data={weekly} options={barOptions} />}
          </ChartCard>
        </>
      )}

      {page === "patterns" && (
        <>
          <ChartCard
            title="Most active hours of day"
            subtitle="Net crossings by clock hour (UTC), last 60 days."
          >
            {popularTotal && <Bar data={popularTotal} options={barOptions} />}
          </ChartCard>

          <ChartCard
            title="Peak direction by hour"
            subtitle="Same window: in vs out per hour of day (UTC), stacked."
          >
            {popularSplit && (
              <Bar data={popularSplit} options={barStackedOptions} />
            )}
          </ChartCard>
        </>
      )}
    </section>
  );
}
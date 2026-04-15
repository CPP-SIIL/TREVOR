import type { OccupancyStats, PopularHours, SeriesBlock, Stats } from "./types";

function ensureSeries(raw: unknown): SeriesBlock {
  const s = raw && typeof raw === "object" ? (raw as Record<string, unknown>) : {};
  const labels = Array.isArray(s.labels) ? (s.labels as string[]) : [];
  const inn = Array.isArray(s.in) ? (s.in as number[]) : [];
  const out = Array.isArray(s.out) ? (s.out as number[]) : [];
  return { labels, in: inn, out };
}

function ensurePopular(raw: unknown): PopularHours {
  const base = ensureSeries(raw);
  const p = raw && typeof raw === "object" ? (raw as Record<string, unknown>) : {};
  const total = Array.isArray(p.total) ? (p.total as number[]) : [];
  return { ...base, total };
}

const LABELS_24 = () =>
  Array.from({ length: 24 }, (_, h) => `${String(h).padStart(2, "0")}:00`);

function padNullable24(arr: unknown): (number | null)[] {
  const a = Array.isArray(arr) ? arr : [];
  const out: (number | null)[] = [];
  for (let i = 0; i < 24; i++) {
    const v = a[i];
    if (v === null || v === undefined) out.push(null);
    else if (typeof v === "number" && !Number.isNaN(v)) out.push(v);
    else if (typeof v === "string" && v !== "") {
      const n = Number(v);
      out.push(Number.isNaN(n) ? null : n);
    } else out.push(null);
  }
  return out;
}

function padNum24(arr: unknown, fill = 0): number[] {
  const a = Array.isArray(arr) ? arr : [];
  const out: number[] = [];
  for (let i = 0; i < 24; i++) {
    const v = a[i];
    const n = typeof v === "number" ? v : Number(v);
    out.push(typeof n === "number" && !Number.isNaN(n) ? n : fill);
  }
  return out;
}

function padLabels24(arr: unknown): string[] {
  const base = LABELS_24();
  if (!Array.isArray(arr)) return base;
  return base.map((d, i) => (typeof arr[i] === "string" ? arr[i] : d));
}

function defaultOccupancy(): OccupancyStats {
  return {
    timezone_note: "UTC",
    weekday_label: "—",
    labels_24h: LABELS_24(),
    today_hourly_net: Array.from({ length: 24 }, () => null),
    today_cumulative: Array.from({ length: 24 }, () => null),
    baseline_sample_days: 0,
    baseline_hourly_net_avg: Array.from({ length: 24 }, () => 0),
    baseline_cumulative: Array.from({ length: 24 }, () => 0),
  };
}

function ensureOccupancy(raw: unknown): OccupancyStats {
  if (!raw || typeof raw !== "object") {
    return defaultOccupancy();
  }
  const o = raw as Record<string, unknown>;
  return {
    timezone_note: typeof o.timezone_note === "string" ? o.timezone_note : "UTC",
    weekday_label: typeof o.weekday_label === "string" ? o.weekday_label : "—",
    labels_24h: padLabels24(o.labels_24h),
    today_hourly_net: padNullable24(o.today_hourly_net),
    today_cumulative: padNullable24(o.today_cumulative),
    baseline_sample_days:
      typeof o.baseline_sample_days === "number" ? o.baseline_sample_days : 0,
    baseline_hourly_net_avg: padNum24(o.baseline_hourly_net_avg, 0),
    baseline_cumulative: padNum24(o.baseline_cumulative, 0),
  };
}

/** Makes API payloads safe for charts (missing fields, old servers, wrong lengths). */
export function normalizeStats(raw: unknown): Stats {
  const r = raw && typeof raw === "object" ? (raw as Record<string, unknown>) : {};
  const totalsRaw =
    r.totals && typeof r.totals === "object" ? (r.totals as Record<string, unknown>) : {};

  return {
    updated_at:
      typeof r.updated_at === "string" ? r.updated_at : new Date().toISOString(),
    totals: {
      in: typeof totalsRaw.in === "number" ? totalsRaw.in : 0,
      out: typeof totalsRaw.out === "number" ? totalsRaw.out : 0,
    },
    hourly: ensureSeries(r.hourly),
    daily: ensureSeries(r.daily),
    weekly: ensureSeries(r.weekly),
    popular_hours: ensurePopular(r.popular_hours),
    occupancy: ensureOccupancy(r.occupancy),
  };
}

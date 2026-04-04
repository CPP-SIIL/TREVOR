const HOURLY_24_KEY = "trevor-hourly-24h";

export function getInitialHourly24h(): boolean {
  try {
    return localStorage.getItem(HOURLY_24_KEY) === "true";
  } catch {
    return false;
  }
}

export function persistHourly24h(value: boolean) {
  try {
    localStorage.setItem(HOURLY_24_KEY, value ? "true" : "false");
  } catch {
    /* ignore */
  }
}

/** Backend sends `HH:MM` UTC bucket starts, e.g. `14:00`. */
export function formatHourlyAxisLabel(label24: string, use24Hour: boolean): string {
  const [hs, ms] = label24.split(":");
  const h = Number(hs);
  const m = Number(ms ?? 0);
  if (Number.isNaN(h)) return label24;
  if (use24Hour) {
    return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
  }
  const ap = h >= 12 ? "pm" : "am";
  const h12 = h % 12 || 12;
  if (m === 0) return `${h12} ${ap}`;
  return `${h12}:${String(m).padStart(2, "0")} ${ap}`;
}

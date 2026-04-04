export type ChartPalette = {
  tick: string;
  grid: string;
  in: string;
  inFill: string;
  out: string;
  outFill: string;
  total: string;
  occupancyLine: string;
  baseline: string;
};

export type ThemeMode = "light" | "dark";

export function getChartPalette(mode: ThemeMode): ChartPalette {
  if (mode === "dark") {
    return {
      tick: "#8b9cb3",
      grid: "rgba(141, 156, 179, 0.12)",
      in: "rgba(61, 214, 140, 0.9)",
      inFill: "rgba(61, 214, 140, 0.15)",
      out: "rgba(110, 182, 255, 0.92)",
      outFill: "rgba(110, 182, 255, 0.12)",
      total: "rgba(196, 167, 255, 0.88)",
      occupancyLine: "rgba(94, 234, 212, 0.95)",
      baseline: "rgba(251, 191, 36, 0.9)",
    };
  }
  return {
    tick: "#4b5565",
    grid: "rgba(15, 23, 42, 0.08)",
    in: "rgba(4, 120, 87, 0.92)",
    inFill: "rgba(4, 120, 87, 0.12)",
    out: "rgba(37, 99, 235, 0.9)",
    outFill: "rgba(37, 99, 235, 0.1)",
    total: "rgba(91, 33, 182, 0.85)",
    occupancyLine: "rgba(13, 148, 136, 0.95)",
    baseline: "rgba(180, 83, 9, 0.9)",
  };
}

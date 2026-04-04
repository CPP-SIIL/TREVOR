export type SeriesBlock = {
  labels: string[];
  in: number[];
  out: number[];
};

export type PopularHours = SeriesBlock & {
  total: number[];
};

export type OccupancyStats = {
  timezone_note: string;
  weekday_label: string;
  labels_24h: string[];
  today_hourly_net: (number | null)[];
  today_cumulative: (number | null)[];
  baseline_sample_days: number;
  baseline_hourly_net_avg: number[];
  baseline_cumulative: number[];
};

export type Stats = {
  updated_at: string;
  totals: { in: number; out: number };
  hourly: SeriesBlock;
  daily: SeriesBlock;
  weekly: SeriesBlock;
  popular_hours: PopularHours;
  occupancy: OccupancyStats;
};

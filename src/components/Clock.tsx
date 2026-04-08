import { useEffect, useState } from "react";

function formatLocalTime(d: Date) {
  return d.toLocaleString(undefined, {
    weekday: "short",
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
  });
}

export function Clock() {
  const [now, setNow] = useState(() => new Date());

  useEffect(() => {
    const id = window.setInterval(() => setNow(new Date()), 1000);
    return () => window.clearInterval(id);
  }, []);

  return (
    <div className="header__clock" aria-live="polite">
      <span className="header__clock-label">Local time</span>
      <time className="header__clock-value" dateTime={now.toISOString()}>
        {formatLocalTime(now)}
      </time>
    </div>
  );
}

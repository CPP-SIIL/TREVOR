import { normalizeStats } from "./normalizeStats";
import type { Stats } from "./types";

function friendlyFetchError(err: unknown): Error {
  if (!(err instanceof Error)) return new Error("Could not load dashboard data.");
  const m = err.message || "";
  if (m === "Failed to fetch" || m.includes("NetworkError") || m.includes("Load failed")) {
    return new Error(
      "Cannot reach the API. If you use npm run dev, start the server in another terminal: uvicorn dashboard_api:app --host 127.0.0.1 --port 8000"
    );
  }
  return err;
}

export async function fetchStats(): Promise<Stats> {
  let res: Response;
  try {
    res = await fetch("/api/stats");
  } catch (e) {
    throw friendlyFetchError(e);
  }
  if (!res.ok) {
    throw new Error(`API error ${res.status}. Check that dashboard_api is running on port 8000.`);
  }
  const text = await res.text();
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    throw new Error(
      "API did not return JSON (got HTML or an error page). Rebuild the UI (npm run build) and open the app from the same host as the API."
    );
  }
  return normalizeStats(raw);
}

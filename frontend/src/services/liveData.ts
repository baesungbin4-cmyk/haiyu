import { useCallback, useEffect, useMemo, useState } from "react";

import {
  alerts as demoAlerts,
  devices as demoDevices,
  vessels as demoVessels,
  type DeviceStatus,
  type RiskAlert,
  type RiskLevel,
  type TrackPoint,
  type VesselTrack,
} from "./mockData";
import type { RealtimeMessage } from "./realtime";

export type DataMode = "demo" | "live" | "mixed" | "offline";

export interface SensorReading {
  id: string;
  deviceId: string;
  sensorType: string;
  timestamp: number | null;
  payload: Record<string, unknown>;
}

export interface LiveDashboardData {
  alerts: RiskAlert[];
  apiStatus: "idle" | "online" | "offline";
  apiToken: string;
  deviceOnlineLabel: string;
  devices: DeviceStatus[];
  mode: DataMode;
  lastUpdated: Date | null;
  highestRisk: string;
  refresh: () => Promise<void>;
  sensorReadings: SensorReading[];
  setApiToken: (value: string) => void;
  vessels: VesselTrack[];
}

interface DashboardLiveState {
  alerts: RiskAlert[];
  apiStatus: LiveDashboardData["apiStatus"];
  devices: DeviceStatus[];
  mode: DataMode;
  lastUpdated: Date | null;
  sensorReadings: SensorReading[];
  vessels: VesselTrack[];
}

const DEFAULT_API_BASE_URL = "http://localhost:8000";
const POLL_INTERVAL_MS = 5_000;
const RISK_ORDER: RiskLevel[] = ["none", "low", "medium", "high"];

export function useLiveDashboardData(
  lastMessage: RealtimeMessage | null,
): LiveDashboardData {
  const [apiToken, setApiTokenState] = useState(readStoredToken);
  const [state, setState] = useState<DashboardLiveState>(() => demoState("idle"));

  const setApiToken = useCallback((value: string) => {
    setApiTokenState(value);
    window.localStorage.setItem("apiToken", value);
  }, []);

  const refresh = useCallback(async () => {
    try {
      const [deviceRows, alarmRows, trackRows, sensorRows] = await Promise.all([
        fetchApiRows("/api/v1/devices", apiToken),
        fetchApiRows("/api/v1/alarms?limit=100", apiToken),
        fetchApiRows("/api/v1/tracks?limit=100", apiToken),
        fetchApiRows("/api/v1/sensors?limit=100", apiToken),
      ]);
      const liveDevices = deviceRows.map(deviceFromApi);
      const liveAlerts = alarmRows.map(alertFromApi);
      const liveSensors = sensorRows.map(sensorFromApi);
      const liveVessels = vesselsFromTracks(trackRows, liveAlerts);
      setState(
        mergedState({
          alerts: liveAlerts,
          apiStatus: "online",
          devices: liveDevices,
          sensorReadings: liveSensors,
          vessels: liveVessels,
        }),
      );
    } catch {
      setState(demoState("offline"));
    }
  }, [apiToken]);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), POLL_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [refresh]);

  useEffect(() => {
    if (lastMessage === null) {
      return;
    }
    setState((current) => applyRealtimeMessage(current, lastMessage));
  }, [lastMessage]);

  return useMemo(
    () => ({
      ...state,
      apiToken,
      deviceOnlineLabel: onlineLabel(state.devices),
      highestRisk: highestRiskLabel(state.alerts),
      refresh,
      setApiToken,
    }),
    [apiToken, refresh, setApiToken, state],
  );
}

function readStoredToken() {
  return window.localStorage.getItem("apiToken") ?? "";
}

function demoState(apiStatus: LiveDashboardData["apiStatus"]): DashboardLiveState {
  return {
    alerts: demoAlerts,
    apiStatus,
    devices: demoDevices,
    mode: "demo" as DataMode,
    lastUpdated: null,
    sensorReadings: [] as SensorReading[],
    vessels: demoVessels,
  };
}

function mergedState(value: {
  alerts: RiskAlert[];
  apiStatus: LiveDashboardData["apiStatus"];
  devices: DeviceStatus[];
  sensorReadings: SensorReading[];
  vessels: VesselTrack[];
}) {
  const hasLive =
    value.alerts.length > 0 ||
    value.devices.length > 0 ||
    value.sensorReadings.length > 0 ||
    value.vessels.length > 0;
  const usesFallback =
    value.alerts.length === 0 || value.devices.length === 0 || value.vessels.length === 0;
  const mode: DataMode = hasLive && usesFallback ? "mixed" : hasLive ? "live" : "demo";
  return {
    alerts: value.alerts.length > 0 ? value.alerts : demoAlerts,
    apiStatus: value.apiStatus,
    devices: value.devices.length > 0 ? value.devices : demoDevices,
    mode,
    lastUpdated: new Date(),
    sensorReadings: value.sensorReadings,
    vessels: value.vessels.length > 0 ? value.vessels : demoVessels,
  };
}

async function fetchApiRows(path: string, token: string) {
  const response = await window.fetch(`${apiBaseUrl()}${path}`, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
  if (!response.ok) {
    throw new Error(`API ${response.status}`);
  }
  const payload: unknown = await response.json();
  return Array.isArray(payload) ? payload.filter(isRecord) : [];
}

function apiBaseUrl() {
  const env = import.meta.env.VITE_API_BASE_URL || DEFAULT_API_BASE_URL;
  return String(env).replace(/\/$/, "");
}

function applyRealtimeMessage(
  current: DashboardLiveState,
  message: RealtimeMessage,
): DashboardLiveState {
  const type = eventType(message);
  const data = eventData(message);
  if (type === "track") {
    return {
      ...current,
      mode: current.mode === "live" ? "live" : "mixed",
      lastUpdated: new Date(),
      vessels: upsertVessel(current.vessels, vesselFromTrackRow(data, current.alerts)),
    };
  }
  if (type === "alert" || type === "alarm") {
    const alert = alertFromApi(data);
    return {
      ...current,
      alerts: upsertAlert(current.alerts, alert),
      mode: current.mode === "live" ? "live" : "mixed",
      lastUpdated: new Date(),
    };
  }
  if (type === "status" || type === "lwt") {
    return {
      ...current,
      devices: upsertDevice(current.devices, deviceFromApi(data)),
      mode: current.mode === "live" ? "live" : "mixed",
      lastUpdated: new Date(),
    };
  }
  if (type === "sensor") {
    return {
      ...current,
      mode: current.mode === "live" ? "live" : "mixed",
      lastUpdated: new Date(),
      sensorReadings: [sensorFromApi(data), ...current.sensorReadings].slice(0, 50),
    };
  }
  return current;
}

function deviceFromApi(row: Record<string, unknown>): DeviceStatus {
  const id = stringValue(row.device_id) || stringValue(row.deviceId) || "unknown";
  return {
    id,
    name: stringValue(row.name) || id,
    berth: stringValue(row.berth) || "实时接入",
    online: booleanValue(row.online),
    cpu: percentValue(row.cpu),
    memory: percentValue(row.mem ?? row.memory),
    npu: percentValue(row.npu),
    latencyMs: numberValue(row.latency_ms ?? row.latencyMs) ?? 0,
  };
}

function alertFromApi(row: Record<string, unknown>): RiskAlert {
  const deviceId = stringValue(row.device_id) || stringValue(row.deviceId) || "unknown";
  const timestamp = numberValue(row.timestamp);
  const vesselId =
    stringValue(row.track_id) || stringValue(row.trackId) || stringValue(row.vesselId) || deviceId;
  return {
    id:
      stringValue(row.id) ||
      `A-${deviceId}-${stringValue(row.seq) || String(timestamp ?? Date.now())}`,
    time: timestamp === null ? "--:--:--" : new Date(timestamp * 1000).toISOString().slice(11, 19),
    vesselId,
    deviceId,
    level: riskLevel(row.risk_level ?? row.level),
    tcpaSeconds: numberValue(row.tcpa ?? row.tcpa_seconds),
    cpaMeters: numberValue(row.cpa ?? row.cpa_distance_m ?? row.dist),
    message: stringValue(row.message ?? row.msg) || "实时告警事件",
  };
}

function sensorFromApi(row: Record<string, unknown>): SensorReading {
  const payload = isRecord(row.payload) ? row.payload : row;
  const deviceId = stringValue(row.device_id) || stringValue(row.deviceId) || "unknown";
  const sensorType =
    stringValue(row.sensor_type) || stringValue(row.sensorType) || stringValue(payload.type) || "unknown";
  const timestamp = numberValue(row.timestamp);
  return {
    id:
      stringValue(row.id) ||
      `S-${deviceId}-${sensorType}-${stringValue(row.seq) || String(timestamp ?? Date.now())}`,
    deviceId,
    sensorType,
    timestamp,
    payload,
  };
}

function vesselsFromTracks(
  rows: Record<string, unknown>[],
  alertRows: RiskAlert[],
) {
  const grouped = new Map<string, Record<string, unknown>[]>();
  rows.forEach((row) => {
    const id = stringValue(row.track_id) || stringValue(row.trackId) || "unknown";
    const deviceId = stringValue(row.device_id) || stringValue(row.deviceId) || "unknown";
    const key = `${deviceId}:${id}`;
    grouped.set(key, [...(grouped.get(key) ?? []), row]);
  });
  return Array.from(grouped.entries()).map(([key, trackRows]) => {
    const [deviceId, trackId] = key.split(":");
    return vesselFromRows(deviceId, trackId, trackRows, alertRows);
  });
}

function vesselFromRows(
  deviceId: string,
  trackId: string,
  rows: Record<string, unknown>[],
  alertRows: RiskAlert[],
): VesselTrack {
  const points = rows.map(pointFromApi).filter((point): point is TrackPoint => point !== null);
  const position = points[points.length - 1] ?? { x: 50, y: 50 };
  const alert = alertRows.find((item) => item.vesselId === trackId);
  const latest = rows[rows.length - 1] ?? {};
  return {
    id: trackId,
    name: trackId,
    className: stringValue(latest.class_name) || stringValue(latest.cls) || "unknown",
    deviceId,
    confidence: numberValue(latest.confidence ?? latest.conf) ?? 0,
    riskLevel: alert?.level ?? "none",
    speedKnots: numberValue(latest.speed_knots) ?? 0,
    courseDeg: numberValue(latest.course_deg) ?? 0,
    tcpaSeconds: alert?.tcpaSeconds ?? null,
    cpaMeters: alert?.cpaMeters ?? null,
    position,
    history: points.length > 0 ? points : [position],
    prediction: [],
  };
}

function vesselFromTrackRow(
  row: Record<string, unknown>,
  alertRows: RiskAlert[],
) {
  const deviceId = stringValue(row.device_id) || stringValue(row.deviceId) || "unknown";
  const trackId = stringValue(row.track_id) || stringValue(row.trackId) || "unknown";
  return vesselFromRows(deviceId, trackId, [row], alertRows);
}

function pointFromApi(row: Record<string, unknown>) {
  const x = numberValue(row.x);
  const y = numberValue(row.y);
  if (x === null || y === null) {
    return null;
  }
  return { x: clampMap(x), y: clampMap(y) };
}

function upsertVessel(items: VesselTrack[], item: VesselTrack) {
  const existing = items.find((value) => value.id === item.id);
  if (!existing) {
    return [item, ...items];
  }
  return items.map((value) =>
    value.id === item.id
      ? { ...item, history: [...existing.history, item.position].slice(-12) }
      : value,
  );
}

function upsertAlert(items: RiskAlert[], item: RiskAlert) {
  return [item, ...items.filter((value) => value.id !== item.id)].slice(0, 50);
}

function upsertDevice(items: DeviceStatus[], item: DeviceStatus) {
  const exists = items.some((value) => value.id === item.id);
  if (!exists) {
    return [item, ...items];
  }
  return items.map((value) => (value.id === item.id ? { ...value, ...item } : value));
}

function eventType(message: RealtimeMessage) {
  return stringValue(message.type || message.event_type || message.eventType).toLowerCase();
}

function eventData(message: RealtimeMessage) {
  if (isRecord(message.data)) {
    return message.data;
  }
  if (isRecord(message.payload)) {
    return message.payload;
  }
  return message;
}

function onlineLabel(devices: DeviceStatus[]) {
  const online = devices.filter((device) => device.online).length;
  return `${online}/${devices.length}`;
}

function highestRiskLabel(alerts: RiskAlert[]) {
  const highest = alerts.reduce<RiskLevel>((current, alert) => {
    return RISK_ORDER.indexOf(alert.level) > RISK_ORDER.indexOf(current)
      ? alert.level
      : current;
  }, "none");
  return { high: "高", medium: "中", low: "低", none: "无" }[highest];
}

function riskLevel(value: unknown): RiskLevel {
  const normalized = stringValue(value).toLowerCase();
  if (normalized === "high" || normalized === "medium" || normalized === "low") {
    return normalized;
  }
  return "none";
}

function percentValue(value: unknown) {
  const number = numberValue(value);
  if (number === null) {
    return 0;
  }
  return Math.round(number <= 1 ? number * 100 : number);
}

function numberValue(value: unknown) {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function booleanValue(value: unknown) {
  if (typeof value === "boolean") {
    return value;
  }
  const normalized = stringValue(value).toLowerCase();
  return ["1", "ok", "online", "true", "up", "yes"].includes(normalized);
}

function stringValue(value: unknown) {
  if (typeof value === "string") {
    return value.trim();
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return "";
}

function clampMap(value: number) {
  return Math.max(2, Math.min(98, value));
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

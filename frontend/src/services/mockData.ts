export const PROTOTYPE_LABEL = "样机级演示数据，未经实港验证";

export type RiskLevel = "none" | "low" | "medium" | "high";

export interface DeviceStatus {
  id: string;
  name: string;
  berth: string;
  online: boolean;
  cpu: number;
  memory: number;
  npu: number;
  latencyMs: number;
}

export interface TrackPoint {
  x: number;
  y: number;
}

export interface VesselTrack {
  id: string;
  name: string;
  className: string;
  deviceId: string;
  confidence: number;
  riskLevel: RiskLevel;
  speedKnots: number;
  courseDeg: number;
  tcpaSeconds: number | null;
  cpaMeters: number | null;
  position: TrackPoint;
  history: TrackPoint[];
  prediction: TrackPoint[];
}

export interface RiskAlert {
  id: string;
  time: string;
  vesselId: string;
  deviceId: string;
  level: RiskLevel;
  tcpaSeconds: number | null;
  cpaMeters: number | null;
  message: string;
}

export interface TrendPoint {
  time: string;
  alerts: number;
  highRisk: number;
  avgLatencyMs: number;
}

export interface SystemMetric {
  label: string;
  value: string;
  delta: string;
  tone: "cyan" | "green" | "amber" | "red";
}

export const devices: DeviceStatus[] = [
  {
    id: "edge-01",
    name: "东引桥摄像终端",
    berth: "A1-A3",
    online: true,
    cpu: 42,
    memory: 61,
    npu: 78,
    latencyMs: 286,
  },
  {
    id: "edge-02",
    name: "北泊位弱光终端",
    berth: "B2",
    online: true,
    cpu: 37,
    memory: 54,
    npu: 71,
    latencyMs: 304,
  },
  {
    id: "edge-03",
    name: "闸口航道终端",
    berth: "C1",
    online: false,
    cpu: 0,
    memory: 0,
    npu: 0,
    latencyMs: 0,
  },
];

export const vessels: VesselTrack[] = [
  {
    id: "T-104",
    name: "货船 104",
    className: "cargo",
    deviceId: "edge-01",
    confidence: 0.91,
    riskLevel: "high",
    speedKnots: 5.2,
    courseDeg: 274,
    tcpaSeconds: 22,
    cpaMeters: 38,
    position: { x: 62, y: 42 },
    history: [
      { x: 82, y: 36 },
      { x: 76, y: 38 },
      { x: 70, y: 40 },
      { x: 66, y: 41 },
      { x: 62, y: 42 },
    ],
    prediction: [
      { x: 58, y: 43 },
      { x: 54, y: 45 },
      { x: 50, y: 47 },
      { x: 46, y: 49 },
    ],
  },
  {
    id: "T-221",
    name: "拖轮 221",
    className: "tug",
    deviceId: "edge-02",
    confidence: 0.88,
    riskLevel: "medium",
    speedKnots: 3.8,
    courseDeg: 96,
    tcpaSeconds: 54,
    cpaMeters: 86,
    position: { x: 33, y: 68 },
    history: [
      { x: 21, y: 72 },
      { x: 25, y: 71 },
      { x: 29, y: 69 },
      { x: 33, y: 68 },
    ],
    prediction: [
      { x: 38, y: 66 },
      { x: 42, y: 64 },
      { x: 46, y: 62 },
    ],
  },
  {
    id: "T-309",
    name: "巡检船 309",
    className: "inspection",
    deviceId: "edge-01",
    confidence: 0.84,
    riskLevel: "low",
    speedKnots: 2.1,
    courseDeg: 12,
    tcpaSeconds: 118,
    cpaMeters: 132,
    position: { x: 72, y: 74 },
    history: [
      { x: 70, y: 86 },
      { x: 70, y: 82 },
      { x: 71, y: 78 },
      { x: 72, y: 74 },
    ],
    prediction: [
      { x: 73, y: 70 },
      { x: 74, y: 66 },
      { x: 75, y: 62 },
    ],
  },
];

export const alerts: RiskAlert[] = [
  {
    id: "A-9007",
    time: "14:57:08",
    vesselId: "T-104",
    deviceId: "edge-01",
    level: "high",
    tcpaSeconds: 22,
    cpaMeters: 38,
    message: "目标进入联合阈值区",
  },
  {
    id: "A-9006",
    time: "14:54:42",
    vesselId: "T-221",
    deviceId: "edge-02",
    level: "medium",
    tcpaSeconds: 54,
    cpaMeters: 86,
    message: "航向靠近泊位外缘",
  },
  {
    id: "A-9005",
    time: "14:50:19",
    vesselId: "T-309",
    deviceId: "edge-01",
    level: "low",
    tcpaSeconds: 118,
    cpaMeters: 132,
    message: "低速巡检轨迹稳定",
  },
];

export const trendPoints: TrendPoint[] = [
  { time: "09:00", alerts: 2, highRisk: 0, avgLatencyMs: 301 },
  { time: "10:00", alerts: 5, highRisk: 1, avgLatencyMs: 294 },
  { time: "11:00", alerts: 4, highRisk: 1, avgLatencyMs: 287 },
  { time: "12:00", alerts: 7, highRisk: 2, avgLatencyMs: 306 },
  { time: "13:00", alerts: 6, highRisk: 1, avgLatencyMs: 298 },
  { time: "14:00", alerts: 9, highRisk: 3, avgLatencyMs: 286 },
];

export const systemMetrics: SystemMetric[] = [
  { label: "在线设备", value: "12/14", delta: "MQTT QoS1", tone: "cyan" },
  { label: "当前最高风险", value: "高", delta: "TCPA 22s", tone: "red" },
  { label: "24H告警", value: "37", delta: "+8 较前日", tone: "amber" },
  { label: "端到端延迟", value: "286ms", delta: "样机链路", tone: "green" },
  { label: "离线缓存", value: "0/128", delta: "待补传", tone: "green" },
];

export const riskLabel: Record<RiskLevel, string> = {
  none: "无",
  low: "低",
  medium: "中",
  high: "高",
};

export const dashboardSummary = {
  title: "运行监控",
  deviceOnline: "12/14",
  highestRisk: "高",
  alertCount24h: 37,
  ws: "degraded",
  mqtt: "degraded",
  edge: "degraded",
};

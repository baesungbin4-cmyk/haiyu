import { useEffect, useState } from "react";

import { PROTOTYPE_LABEL } from "../services/mockData";
import type { ConnectionStatus } from "../services/realtime";

interface StatusBarProps {
  summary: {
    title: string;
    deviceOnline: string;
    highestRisk: string;
    alertCount24h: number;
    ws: ConnectionStatus;
    mqtt: ConnectionStatus;
    edge: ConnectionStatus;
  };
}

function StatusPill({
  label,
  status,
}: {
  label: string;
  status: ConnectionStatus;
}) {
  return (
    <span className={`status-pill status-${status}`}>
      <span className="status-pill__dot" />
      {label}
    </span>
  );
}

export default function StatusBar({ summary }: StatusBarProps) {
  const [now, setNow] = useState(() => new Date());

  useEffect(() => {
    const timer = window.setInterval(() => setNow(new Date()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  return (
    <header className="status-bar">
      <div className="brand-lockup">
        <span className="status-brand-mark">HY</span>
        <div>
          <strong>海御·智慧物联网安全平台</strong>
          <span>{summary.title}</span>
        </div>
      </div>
      <div className="status-bar__metrics" aria-label="global summary">
        <span>在线 {summary.deviceOnline}</span>
        <span>风险 {summary.highestRisk}</span>
        <span>告警 {summary.alertCount24h}</span>
      </div>
      <div className="status-bar__links">
        <StatusPill label="WS" status={summary.ws} />
        <StatusPill label="MQTT" status={summary.mqtt} />
        <StatusPill label="边缘" status={summary.edge} />
      </div>
      <time>{now.toLocaleString("zh-CN", { hour12: false })}</time>
      <span className="prototype-badge">{PROTOTYPE_LABEL}</span>
    </header>
  );
}

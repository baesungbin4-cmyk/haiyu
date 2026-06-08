import { useMemo, useState } from "react";

import {
  riskLabel,
  type RiskAlert,
  type VesselTrack,
} from "../services/mockData";

interface RiskPanelProps {
  alerts: RiskAlert[];
  selectedVessel: VesselTrack;
  selectedAlertId: string;
  onSelect: (vesselId: string) => void;
}

export default function RiskPanel({
  alerts,
  selectedAlertId,
  selectedVessel,
  onSelect,
}: RiskPanelProps) {
  const [activeAlertId, setActiveAlertId] = useState(selectedAlertId);
  const activeAlert = useMemo(
    () => alerts.find((alert) => alert.id === activeAlertId) ?? alerts[0],
    [activeAlertId, alerts]
  );

  return (
    <aside className="right-intelligence-column">
      <section className="intel-card cpa-radar-card">
        <span className="micro-label">CPA / TCPA RADAR</span>
        <div className="radar-analysis">
          <svg aria-hidden="true" viewBox="0 0 120 92">
            <circle cx="46" cy="46" r="36" />
            <circle cx="46" cy="46" r="22" />
            <circle cx="46" cy="46" r="9" />
            <path d="M46 46 L83 27" />
            <path className="radar-risk" d="M46 46 C61 36 75 39 91 55" />
          </svg>
          <dl>
            <div>
              <dt>CPA</dt>
              <dd>{selectedVessel.cpaMeters ?? "-"}m</dd>
            </div>
            <div>
              <dt>TCPA</dt>
              <dd>{selectedVessel.tcpaSeconds ?? "-"}s</dd>
            </div>
            <div>
              <dt>CONF</dt>
              <dd>{Math.round(selectedVessel.confidence * 100)}%</dd>
            </div>
          </dl>
        </div>
      </section>

      <section className="intel-card threat-card">
        <span className="micro-label">THREAT RANKING</span>
        {alerts.map((alert, index) => (
          <button
            className={`threat-row risk-${alert.level}`}
            key={alert.id}
            onClick={() => {
              setActiveAlertId(alert.id);
              onSelect(alert.vesselId);
            }}
            type="button"
          >
            <span>{String(index + 1).padStart(2, "0")}</span>
            <strong>{alert.vesselId}</strong>
            <em>{riskLabel[alert.level]}</em>
          </button>
        ))}
      </section>

      <section className="intel-card anomaly-card">
        <span className="micro-label">AI ANOMALY</span>
        <strong>{activeAlert.message}</strong>
        <p>{activeAlert.deviceId} / prediction branch 02 / ETA replay available</p>
        <div className="action-grid">
          <button type="button">Investigate</button>
          <button type="button">Assign</button>
          <button type="button">Replay</button>
          <button type="button">Resolve</button>
        </div>
      </section>

      <section className="intel-card confidence-card">
        <span className="micro-label">PREDICTION CONFIDENCE</span>
        <div className="confidence-bar">
          <i style={{ width: `${Math.round(selectedVessel.confidence * 100)}%` }} />
        </div>
        <strong>{Math.round(selectedVessel.confidence * 100)}%</strong>
      </section>
    </aside>
  );
}

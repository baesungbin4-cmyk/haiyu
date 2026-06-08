import { riskLabel, type RiskAlert } from "../services/mockData";
import { useVirtualWindow } from "./useVirtualWindow";

interface AlertTableProps {
  alerts: RiskAlert[];
  selectedVesselId: string;
  onSelect: (vesselId: string) => void;
}

const ALERT_ROW_HEIGHT = 64;
const ALERT_VIEWPORT_HEIGHT = 260;

export default function AlertTable({
  alerts,
  selectedVesselId,
  onSelect,
}: AlertTableProps) {
  const virtual = useVirtualWindow(alerts, {
    rowHeight: ALERT_ROW_HEIGHT,
    viewportHeight: ALERT_VIEWPORT_HEIGHT,
  });

  return (
    <>
      <div className="alert-table" onScroll={virtual.onScroll} role="list">
        <div className="virtual-spacer" style={{ height: virtual.topPad }} />
        {virtual.items.map((alert) => (
          <button
            className={`alert-row risk-${alert.level}`}
            data-selected={alert.vesselId === selectedVesselId}
            key={alert.id}
            onClick={() => onSelect(alert.vesselId)}
            role="listitem"
            type="button"
          >
            <span className="alert-row__time">{alert.time}</span>
            <span>
              <strong>{alert.vesselId}</strong>
              <small>{alert.message}</small>
            </span>
            <span className="alert-row__risk">{riskLabel[alert.level]}</span>
            <span className="alert-row__numbers">
              TCPA {alert.tcpaSeconds ?? "-"}s
              <br />
              CPA {alert.cpaMeters ?? "-"}m
            </span>
          </button>
        ))}
        <div
          className="virtual-spacer"
          style={{ height: virtual.bottomPad }}
        />
      </div>
      <div className="window-note">可见 {virtual.items.length}/{alerts.length}</div>
    </>
  );
}

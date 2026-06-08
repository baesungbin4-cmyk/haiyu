import {
  riskLabel,
  type VesselTrack,
} from "../services/mockData";
import { useVirtualWindow } from "./useVirtualWindow";

interface VesselListProps {
  vessels: VesselTrack[];
  selectedId: string;
  onSelect: (id: string) => void;
}

const VESSEL_ROW_HEIGHT = 54;
const VESSEL_VIEWPORT_HEIGHT = 300;

export default function VesselList({
  vessels,
  selectedId,
  onSelect,
}: VesselListProps) {
  const virtual = useVirtualWindow(vessels, {
    rowHeight: VESSEL_ROW_HEIGHT,
    viewportHeight: VESSEL_VIEWPORT_HEIGHT,
  });

  return (
    <section className="panel vessel-list">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">Targets</span>
          <h2>船舶目标</h2>
        </div>
      </div>
      <div
        className="vessel-list__scroll"
        onScroll={virtual.onScroll}
        role="list"
      >
        <div className="virtual-spacer" style={{ height: virtual.topPad }} />
        {virtual.items.map((vessel) => (
          <button
            className={`vessel-row risk-${vessel.riskLevel}`}
            data-selected={vessel.id === selectedId}
            key={vessel.id}
            onClick={() => onSelect(vessel.id)}
            role="listitem"
            type="button"
          >
            <span className="risk-dot" />
            <span>
              <strong>{vessel.name}</strong>
              <small>
                {vessel.id} · {vessel.className} · {vessel.deviceId}
              </small>
            </span>
            <em>{riskLabel[vessel.riskLevel]}</em>
          </button>
        ))}
        <div
          className="virtual-spacer"
          style={{ height: virtual.bottomPad }}
        />
      </div>
      <div className="window-note">
        可见 {virtual.items.length}/{vessels.length}
      </div>
    </section>
  );
}

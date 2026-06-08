import { useMemo, useState, type MouseEvent } from "react";

import {
  riskLabel,
  type DeviceStatus,
  type RiskAlert,
  type VesselTrack,
} from "../services/mockData";

interface PortSituationPanelProps {
  alerts: RiskAlert[];
  devices: DeviceStatus[];
  highlight: string | null;
  vessels: VesselTrack[];
  selectedId: string;
  onSelect: (id: string) => void;
}

type LayerKey = "ais" | "track" | "heat" | "camera" | "edge" | "berth" | "fairway";

interface ContextMenuState {
  vesselId: string;
  x: number;
  y: number;
}

const devicePositions = [
  { x: 18, y: 28 },
  { x: 80, y: 58 },
  { x: 48, y: 82 },
];
const gridTicks = [10, 20, 30, 40, 50, 60, 70, 80, 90];
const rangeRings = [10, 18, 27, 36];
const contours = [
  "M5 73 C21 65 31 58 47 58 C62 58 71 51 90 42",
  "M8 86 C22 78 39 71 54 70 C69 69 77 61 96 55",
  "M13 15 C30 26 43 31 58 28 C72 25 83 29 96 37",
];
const mapLabels = [
  { label: "ANCHORAGE", x: 41, y: 26 },
  { label: "FAIRWAY 03", x: 43, y: 54 },
  { label: "BASIN A", x: 14, y: 42 },
  { label: "TERMINAL B", x: 76, y: 49 },
];
const telemetryDots = Array.from({ length: 96 }, (_, index) => ({
  x: (index * 37 + 9) % 100,
  y: (index * 53 + 17) % 100,
  tone: index % 11 === 0 ? "hot" : index % 5 === 0 ? "warn" : "calm",
}));
const layerLabels: Record<LayerKey, string> = {
  ais: "AIS",
  berth: "Berth",
  camera: "Camera",
  edge: "Edge",
  fairway: "Fairway",
  heat: "Heat",
  track: "Track",
};

function smoothPath(points: { x: number; y: number }[]) {
  if (points.length < 2) {
    return "";
  }

  let path = `M${points[0].x} ${points[0].y}`;
  for (let index = 0; index < points.length - 1; index += 1) {
    const current = points[index];
    const next = points[index + 1];
    const previous = points[index - 1] ?? current;
    const afterNext = points[index + 2] ?? next;
    const controlA = {
      x: current.x + (next.x - previous.x) / 6,
      y: current.y + (next.y - previous.y) / 6,
    };
    const controlB = {
      x: next.x - (afterNext.x - current.x) / 6,
      y: next.y - (afterNext.y - current.y) / 6,
    };
    path += ` C${controlA.x.toFixed(1)} ${controlA.y.toFixed(1)}, ${controlB.x.toFixed(1)} ${controlB.y.toFixed(1)}, ${next.x} ${next.y}`;
  }
  return path;
}

function predictionConePath(vessel: VesselTrack) {
  const future = vessel.prediction[vessel.prediction.length - 1] ?? vessel.position;
  const mid = vessel.prediction[Math.max(0, Math.floor(vessel.prediction.length / 2))] ?? future;
  return [
    `M${vessel.position.x} ${vessel.position.y}`,
    `C${mid.x - 8} ${mid.y - 8}, ${future.x - 10} ${future.y - 6}, ${future.x - 5} ${future.y - 3}`,
    `L${future.x + 8} ${future.y + 7}`,
    `C${future.x - 2} ${future.y + 6}, ${mid.x + 6} ${mid.y + 4}, ${vessel.position.x} ${vessel.position.y}`,
    "Z",
  ].join(" ");
}

export default function PortSituationPanel({
  alerts,
  devices,
  highlight,
  vessels,
  selectedId,
  onSelect,
}: PortSituationPanelProps) {
  const [layers, setLayers] = useState<Record<LayerKey, boolean>>({
    ais: true,
    berth: true,
    camera: true,
    edge: true,
    fairway: true,
    heat: true,
    track: true,
  });
  const [focusMode, setFocusMode] = useState(true);
  const [timeCursor, setTimeCursor] = useState(72);
  const [hoveredVesselId, setHoveredVesselId] = useState<string | null>(null);
  const [contextMenu, setContextMenu] = useState<ContextMenuState | null>(null);
  const selectedVessel = useMemo(
    () => vessels.find((vessel) => vessel.id === selectedId) ?? vessels[0],
    [selectedId, vessels]
  );
  const selectedAlert = alerts.find((alert) => alert.vesselId === selectedVessel.id);
  const predictionStep = Math.max(1, Math.round((timeCursor / 100) * 4));

  const openContextMenu = (
    event: MouseEvent<SVGGElement>,
    vesselId: string
  ) => {
    event.preventDefault();
    onSelect(vesselId);
    setContextMenu({ vesselId, x: event.clientX, y: event.clientY });
  };

  return (
    <section className={`gis-module ${focusMode ? "route-focus" : ""} ${highlight ? `highlight-${highlight}` : ""}`}>
      <div className="gis-module-header">
        <div>
          <span>IOT GIS</span>
          <strong>AI Trajectory Prediction Layer</strong>
        </div>
        <div className="gis-toolbar">
          <button className={focusMode ? "is-active" : ""} onClick={() => setFocusMode((value) => !value)} type="button">
            FOCUS
          </button>
          <input
            aria-label="ETA playback"
            max="100"
            min="0"
            onChange={(event) => setTimeCursor(Number(event.target.value))}
            type="range"
            value={timeCursor}
          />
          <output>ETA {Math.round(timeCursor * 0.9)}s</output>
        </div>
      </div>

      <div className="layer-toggle-row">
        {(Object.keys(layerLabels) as LayerKey[]).map((layer) => (
          <button
            aria-pressed={layers[layer]}
            className={layers[layer] ? "is-active" : ""}
            key={layer}
            onClick={() => setLayers((current) => ({ ...current, [layer]: !current[layer] }))}
            type="button"
          >
            {layerLabels[layer]}
          </button>
        ))}
      </div>

      <div className="gis-map-stage" onClick={() => setContextMenu(null)}>
        <svg
          aria-label="IoT tactical GIS trajectory map"
          className="gis-canvas"
          preserveAspectRatio="xMidYMid slice"
          viewBox="0 0 100 100"
        >
          <defs>
            <radialGradient id="seaDepth" cx="52%" cy="48%" r="70%">
              <stop offset="0%" stopColor="#12385f" />
              <stop offset="58%" stopColor="#061a2f" />
              <stop offset="100%" stopColor="#041426" />
            </radialGradient>
            <radialGradient id="heatHigh" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="rgba(94,174,214,0.38)" />
              <stop offset="100%" stopColor="rgba(94,174,214,0)" />
            </radialGradient>
          </defs>
          <rect className="sea-base" height="100" width="100" />
          <g className="geo-grid">
            {gridTicks.map((tick) => (
              <line key={`x-${tick}`} x1={tick} x2={tick} y1="0" y2="100" />
            ))}
            {gridTicks.map((tick) => (
              <line key={`y-${tick}`} x1="0" x2="100" y1={tick} y2={tick} />
            ))}
          </g>
          <g className="range-rings">
            {rangeRings.map((ring) => (
              <circle cx="58" cy="48" key={ring} r={ring} />
            ))}
          </g>
          <g className="contour-overlay">
            {contours.map((contour) => (
              <path d={contour} key={contour} />
            ))}
          </g>
          <g className="telemetry-texture">
            {telemetryDots.map((dot, index) => (
              <circle className={dot.tone} cx={dot.x} cy={dot.y} key={index} r="0.22" />
            ))}
          </g>
          <g className="geo-labels">
            {mapLabels.map((item) => (
              <text key={item.label} x={item.x} y={item.y}>{item.label}</text>
            ))}
          </g>
          <path className="geofence" d="M47 30 L72 38 L78 61 L55 74 L35 62 L34 42 Z" />
          <line className="radar-sweep" x1="58" x2="94" y1="48" y2="31" />
          <path className="coastline" d="M0 0 H30 C22 12 23 25 30 38 C37 51 30 63 17 69 H0 Z" />
          <path className="coastline right" d="M100 21 V100 H75 C82 85 84 70 76 58 C68 45 74 31 100 21 Z" />

          {layers.berth ? (
            <g className="berth-overlay">
              <rect height="11" width="3.5" x="15" y="25" />
              <rect height="16" width="3.5" x="78" y="52" />
              <rect height="4" width="27" x="34" y="80" />
            </g>
          ) : null}

          {layers.fairway ? (
            <g className="fairway-overlay">
              <path d="M7 82 C25 68 43 56 62 42" />
              <path d="M18 24 C39 36 58 55 76 76" />
              <path d="M20 52 C42 48 60 50 85 36" />
              <path d="M10 40 C32 42 55 35 88 25" />
            </g>
          ) : null}

          {layers.heat ? (
            <g className="risk-heat-overlay">
              <ellipse className="heat-high" cx="61" cy="42" rx="17" ry="10" />
              <ellipse className="cyan-zone" cx="71" cy="75" rx="10" ry="6" />
            </g>
          ) : null}

          {layers.camera ? (
            <g className="camera-overlay">
              <path d="M18 28 L39 37 L24 51 Z" />
              <path d="M78 62 L57 53 L72 42 Z" />
              <path d="M48 82 L38 62 L59 65 Z" />
            </g>
          ) : null}

          {layers.edge ? (
            <g className="edge-overlay">
              {devices.map((device, index) => {
                const position = devicePositions[index % devicePositions.length];
                return (
                  <g className={device.online ? "edge-node online" : "edge-node"} key={device.id}>
                    <circle cx={position.x} cy={position.y} r="2" />
                    <text x={position.x + 3} y={position.y + 1}>{device.id}</text>
                  </g>
                );
              })}
            </g>
          ) : null}

          {vessels.map((vessel) => (
            <g
              className={`ais-vessel risk-${vessel.riskLevel}`}
              data-selected={vessel.id === selectedId}
              key={vessel.id}
              onClick={(event) => {
                event.stopPropagation();
                onSelect(vessel.id);
              }}
              onContextMenu={(event) => openContextMenu(event, vessel.id)}
              onMouseEnter={() => setHoveredVesselId(vessel.id)}
              onMouseLeave={() => setHoveredVesselId(null)}
              role="button"
              tabIndex={0}
            >
              {layers.track ? (
                <>
                  <path className="history-line" d={smoothPath(vessel.history)} />
                  <path
                    className="prediction-line"
                    d={smoothPath([
                      vessel.position,
                      ...vessel.prediction.slice(0, predictionStep),
                    ])}
                  />
                  {vessel.id === selectedId ? (
                    <>
                      <path className="prediction-cone" d={predictionConePath(vessel)} />
                      <path
                        className="branch-line"
                        d={smoothPath([
                          vessel.position,
                          { x: vessel.position.x - 5, y: vessel.position.y + 7 },
                          { x: vessel.position.x - 11, y: vessel.position.y + 12 },
                        ])}
                      />
                      <path
                        className="branch-line muted"
                        d={smoothPath([
                          vessel.position,
                          { x: vessel.position.x + 4, y: vessel.position.y + 8 },
                          { x: vessel.position.x + 12, y: vessel.position.y + 10 },
                        ])}
                      />
                      {vessel.prediction.slice(0, predictionStep).map((point, index) => (
                        <circle className="ghost-position" cx={point.x} cy={point.y} key={`${vessel.id}-ghost-${index}`} r={1 + index * 0.22} />
                      ))}
                    </>
                  ) : null}
                </>
              ) : null}
              {layers.ais ? (
                <>
                  <circle className="vessel-pulse" cx={vessel.position.x} cy={vessel.position.y} r="4.8" />
                  <path d={`M${vessel.position.x - 2.5} ${vessel.position.y + 3} L${vessel.position.x} ${vessel.position.y - 3.4} L${vessel.position.x + 2.5} ${vessel.position.y + 3} Z`} />
                  <text x={vessel.position.x + 3.8} y={vessel.position.y - 4}>{vessel.id}</text>
                </>
              ) : null}
            </g>
          ))}
        </svg>

        <div className="map-vignette" />

        <div className="selected-vessel-card">
          <span>SELECTED</span>
          <strong>{selectedVessel.id}</strong>
          <em>{selectedAlert ? selectedAlert.message : "AIS watch"}</em>
        </div>

        {hoveredVesselId ? (
          <div className="telemetry-tooltip">
            <span>{hoveredVesselId}</span>
            <strong>
              {riskLabel[vessels.find((vessel) => vessel.id === hoveredVesselId)?.riskLevel ?? "low"]}
            </strong>
          </div>
        ) : null}

        {contextMenu ? (
          <div
            className="context-menu"
            onClick={(event) => event.stopPropagation()}
            style={{ left: contextMenu.x, top: contextMenu.y }}
          >
            <span>{contextMenu.vesselId}</span>
            <button type="button">INVESTIGATE</button>
            <button type="button">ASSIGN</button>
            <button type="button">REPLAY</button>
            <button type="button">RESOLVE</button>
          </div>
        ) : null}
      </div>
    </section>
  );
}

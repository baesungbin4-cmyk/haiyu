import { useEffect, useMemo, useRef, useState } from "react";

import {
  riskLabel,
  type RiskLevel,
  type VesselTrack,
} from "../services/mockData";
import type { LiveDashboardData, SensorReading } from "../services/liveData";
import type { RealtimeStatus } from "../services/realtime";

type Point = { x: number; y: number };
type LayerKey = "radar" | "heat" | "lanes" | "waypoints";
type PanelKey = "overview" | "map" | "tracks" | "prediction" | "alerts" | "ai";

interface UiVessel {
  id: string;
  name: string;
  deviceId: string;
  x: number;
  y: number;
  heading: number;
  speed: number;
  color: string;
  past: Point[];
  waypoints: Point[];
  altBranches: Point[][];
  source: VesselTrack;
}

interface CanvasMapProps {
  layers: Record<LayerKey, boolean>;
  onSelect: (id: string) => void;
  selectedId: string;
  timeline: number;
  vessels: UiVessel[];
}

const colors = ["#5eaed6", "#8ab4cc", "#c9a84c", "#7ab8d4", "#d8ecf8"];
const riskOrder: RiskLevel[] = ["none", "low", "medium", "high"];

const navItems: Array<{ key: PanelKey; label: string }> = [
  { key: "overview", label: "总览" },
  { key: "map", label: "态势图" },
  { key: "tracks", label: "轨迹" },
  { key: "prediction", label: "预测" },
  { key: "alerts", label: "告警" },
  { key: "ai", label: "边缘 AI" },
];

const layerLabels: Record<LayerKey, string> = {
  heat: "风险热区",
  lanes: "航道线",
  radar: "雷达圈",
  waypoints: "预测点",
};

function useClock() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(new Date()), 33);
    return () => window.clearInterval(timer);
  }, []);
  return now;
}

function clamp01(value: number) {
  return Math.max(0.04, Math.min(0.96, value));
}

function normalized(point: Point) {
  return { x: clamp01(point.x / 100), y: clamp01(point.y / 100) };
}

function vesselWaypoints(vessel: VesselTrack) {
  const base = normalized(vessel.position);
  const prediction = vessel.prediction.length
    ? vessel.prediction.map(normalized)
    : Array.from({ length: 4 }, (_, index) => {
        const rad = ((vessel.courseDeg || 45) - 90) * (Math.PI / 180);
        const step = (index + 1) * 0.045;
        return {
          x: clamp01(base.x + Math.cos(rad) * step),
          y: clamp01(base.y + Math.sin(rad) * step),
        };
      });
  return prediction;
}

function toUiVessels(vessels: VesselTrack[]) {
  return vessels.map((vessel, index) => {
    const position = normalized(vessel.position);
    const waypoints = vesselWaypoints(vessel);
    return {
      id: vessel.id,
      name: vessel.name || vessel.id,
      deviceId: vessel.deviceId,
      x: position.x,
      y: position.y,
      heading: vessel.courseDeg || 0,
      speed: vessel.speedKnots || 0,
      color: colors[index % colors.length],
      past: vessel.history.map(normalized),
      waypoints,
      altBranches: waypoints.length
        ? [
            waypoints.slice(0, 3).map((point, branchIndex) => ({
              x: clamp01(point.x - 0.025 * (branchIndex + 1)),
              y: clamp01(point.y + 0.018 * (branchIndex + 1)),
            })),
          ]
        : [],
      source: vessel,
    } satisfies UiVessel;
  });
}

function viewport(
  canvas: HTMLCanvasElement,
  center: Point,
  zoom: number,
) {
  const cw = Math.max(1, canvas.clientWidth);
  const ch = Math.max(1, canvas.clientHeight);
  const aspect = cw / ch;
  const viewWidth = 1 / zoom;
  const viewHeight = viewWidth / aspect;
  return {
    cw,
    ch,
    viewX: center.x - viewWidth / 2,
    viewY: center.y - viewHeight / 2,
    viewWidth,
    viewHeight,
  };
}

function normToCanvas(point: Point, vp: ReturnType<typeof viewport>) {
  return {
    x: ((point.x - vp.viewX) / vp.viewWidth) * vp.cw,
    y: ((point.y - vp.viewY) / vp.viewHeight) * vp.ch,
  };
}

function canvasToNorm(x: number, y: number, vp: ReturnType<typeof viewport>) {
  return {
    x: (x / vp.cw) * vp.viewWidth + vp.viewX,
    y: (y / vp.ch) * vp.viewHeight + vp.viewY,
  };
}

function currentPosition(vessel: UiVessel, timeline: number) {
  if (timeline < 0.5) {
    const points = [...vessel.past, { x: vessel.x, y: vessel.y }];
    return interpolate(points, timeline * 2);
  }
  if (timeline > 0.5) {
    return interpolate(
      [{ x: vessel.x, y: vessel.y }, ...vessel.waypoints],
      (timeline - 0.5) * 2,
    );
  }
  return { x: vessel.x, y: vessel.y };
}

function interpolate(points: Point[], ratio: number) {
  if (points.length <= 1) {
    return points[0] ?? { x: 0.5, y: 0.5 };
  }
  const scaled = Math.min(points.length - 1, Math.max(0, ratio * (points.length - 1)));
  const index = Math.floor(scaled);
  const next = Math.min(points.length - 1, index + 1);
  const frac = scaled - index;
  return {
    x: points[index].x + (points[next].x - points[index].x) * frac,
    y: points[index].y + (points[next].y - points[index].y) * frac,
  };
}

function drawSmoothPath(ctx: CanvasRenderingContext2D, points: Point[]) {
  if (points.length < 2) {
    return;
  }
  ctx.moveTo(points[0].x, points[0].y);
  for (let index = 0; index < points.length - 1; index += 1) {
    const current = points[index];
    const next = points[index + 1];
    const previous = points[index - 1] ?? current;
    const after = points[index + 2] ?? next;
    const cp1 = {
      x: current.x + (next.x - previous.x) / 6,
      y: current.y + (next.y - previous.y) / 6,
    };
    const cp2 = {
      x: next.x - (after.x - current.x) / 6,
      y: next.y - (after.y - current.y) / 6,
    };
    ctx.bezierCurveTo(cp1.x, cp1.y, cp2.x, cp2.y, next.x, next.y);
  }
}

function drawCanvas(
  canvas: HTMLCanvasElement,
  vessels: UiVessel[],
  selectedId: string,
  timeline: number,
  layers: Record<LayerKey, boolean>,
  center: Point,
  zoom: number,
  time: number,
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return;
  }
  const vp = viewport(canvas, center, zoom);
  canvas.width = vp.cw * 2;
  canvas.height = vp.ch * 2;
  canvas.style.width = `${vp.cw}px`;
  canvas.style.height = `${vp.ch}px`;
  ctx.setTransform(2, 0, 0, 2, 0, 0);
  ctx.clearRect(0, 0, vp.cw, vp.ch);

  const grad = ctx.createRadialGradient(
    vp.cw * 0.5,
    vp.ch * 0.4,
    vp.cw * 0.08,
    vp.cw * 0.5,
    vp.ch * 0.5,
    vp.cw * 1.1,
  );
  grad.addColorStop(0, "#0a2945");
  grad.addColorStop(0.5, "#071e35");
  grad.addColorStop(1, "#030f1c");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, vp.cw, vp.ch);

  ctx.strokeStyle = "rgba(255,255,255,0.035)";
  ctx.lineWidth = 0.5;
  for (let gx = Math.floor(vp.viewX / 0.08) * 0.08; gx <= vp.viewX + vp.viewWidth; gx += 0.08) {
    const px = ((gx - vp.viewX) / vp.viewWidth) * vp.cw;
    ctx.beginPath();
    ctx.moveTo(px, 0);
    ctx.lineTo(px, vp.ch);
    ctx.stroke();
  }
  for (let gy = Math.floor(vp.viewY / 0.08) * 0.08; gy <= vp.viewY + vp.viewHeight; gy += 0.08) {
    const py = ((gy - vp.viewY) / vp.viewHeight) * vp.ch;
    ctx.beginPath();
    ctx.moveTo(0, py);
    ctx.lineTo(vp.cw, py);
    ctx.stroke();
  }

  const radar = normToCanvas({ x: 0.5, y: 0.45 }, vp);
  if (layers.radar) {
    // Pulsing radar center glow
    const radarPulse = 0.4 + 0.15 * Math.sin(time * 0.0025);
    const radarGlow = ctx.createRadialGradient(radar.x, radar.y, 0, radar.x, radar.y, vp.cw * 0.05);
    radarGlow.addColorStop(0, `rgba(94,174,214,${radarPulse})`);
    radarGlow.addColorStop(1, "rgba(94,174,214,0)");
    ctx.fillStyle = radarGlow;
    ctx.beginPath();
    ctx.arc(radar.x, radar.y, vp.cw * 0.05, 0, Math.PI * 2);
    ctx.fill();

    // Breathing concentric rings
    const ringBreathe = 0.5 + 0.5 * Math.sin(time * 0.0018);
    [0.12, 0.24, 0.36, 0.48].forEach((ring, index) => {
      const ringAlpha = (0.06 + index * 0.02) + ringBreathe * 0.03;
      ctx.strokeStyle = `rgba(94,174,214,${ringAlpha})`;
      ctx.lineWidth = 0.6;
      ctx.setLineDash([4, 18]);
      ctx.beginPath();
      ctx.arc(radar.x, radar.y, ring * vp.cw, 0, Math.PI * 2);
      ctx.stroke();
    });
    ctx.setLineDash([]);

    // Radar sweep with afterglow trail
    const angle = (time / 900) % (Math.PI * 2);
    const glowLength = 0.3;
    for (let t = 0; t < 6; t++) {
      const trailAngle = angle - (t * glowLength) / 6;
      const trailAlpha = 0.12 - t * 0.018;
      if (trailAlpha <= 0) continue;
      ctx.strokeStyle = `rgba(94,174,214,${trailAlpha})`;
      ctx.lineWidth = 1.2 + t * 0.3;
      ctx.beginPath();
      ctx.arc(radar.x, radar.y, vp.cw * 0.42, trailAngle - 0.02, trailAngle, false);
      ctx.stroke();
    }

    const sweepX = radar.x + Math.cos(angle) * vp.cw;
    const sweepY = radar.y + Math.sin(angle) * vp.cw;
    const sweep = ctx.createLinearGradient(radar.x, radar.y, sweepX, sweepY);
    sweep.addColorStop(0, "rgba(94,174,214,0.25)");
    sweep.addColorStop(1, "rgba(94,174,214,0)");
    ctx.strokeStyle = sweep;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(radar.x, radar.y);
    ctx.lineTo(sweepX, sweepY);
    ctx.stroke();
  }

  if (layers.heat) {
    const heat = normToCanvas({ x: 0.7, y: 0.55 }, vp);
    const heatGrad = ctx.createRadialGradient(heat.x, heat.y, 0, heat.x, heat.y, vp.cw * 0.18);
    heatGrad.addColorStop(0, "rgba(94,174,214,0.14)");
    heatGrad.addColorStop(1, "rgba(94,174,214,0)");
    ctx.fillStyle = heatGrad;
    ctx.fillRect(0, 0, vp.cw, vp.ch);
  }

  if (layers.lanes) {
    const left = normToCanvas({ x: 0.18, y: 0.52 }, vp);
    const right = normToCanvas({ x: 0.88, y: 0.5 }, vp);
    const cp1 = { x: vp.cw * 0.4, y: vp.ch * 0.45 };
    const cp2 = { x: vp.cw * 0.6, y: vp.ch * 0.4 };
    ctx.strokeStyle = "rgba(94,174,214,0.08)";
    ctx.lineWidth = 8;
    ctx.setLineDash([30, 80]);
    ctx.beginPath();
    ctx.moveTo(left.x, left.y);
    ctx.bezierCurveTo(cp1.x, cp1.y, cp2.x, cp2.y, right.x, right.y);
    ctx.stroke();
    ctx.setLineDash([]);

    // Flowing lane particles along the fairway bezier
    for (let p = 0; p < 4; p++) {
      const t = ((time * 0.015 + p * 0.25) % 1 + 1) % 1; // offset per particle
      const u = 1 - t;
      const bx = u * u * u * left.x + 3 * u * u * t * cp1.x + 3 * u * t * t * cp2.x + t * t * t * right.x;
      const by = u * u * u * left.y + 3 * u * u * t * cp1.y + 3 * u * t * t * cp2.y + t * t * t * right.y;
      const particleAlpha = 0.25 + 0.15 * Math.sin(time * 0.004 + p);
      ctx.fillStyle = `rgba(94,174,214,${particleAlpha})`;
      ctx.shadowColor = "#5eaed6";
      ctx.shadowBlur = 3;
      ctx.beginPath();
      ctx.arc(bx, by, 1.2, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
    }
  }

  vessels.forEach((vessel, vi) => {
    const selected = vessel.id === selectedId;
    const alpha = selected ? 1 : 0.52;
    // Breathing pulse — staggered per vessel for organic feel
    const breathePhase = vi * 0.8;
    const breathe = 0.7 + 0.3 * Math.sin(time * 0.003 + breathePhase);
    const isHighRisk = vessel.source.riskLevel === "high";
    const riskPulse = isHighRisk ? 0.6 + 0.4 * Math.sin(time * 0.005 + breathePhase) : 1;
    const current = currentPosition(vessel, timeline);
    const currentCanvas = normToCanvas(current, vp);
    const pastPoints = [...vessel.past, { x: vessel.x, y: vessel.y }].map((point) =>
      normToCanvas(point, vp),
    );
    const futurePoints = [{ x: current.x, y: current.y }, ...vessel.waypoints].map((point) =>
      normToCanvas(point, vp),
    );

    ctx.globalAlpha = alpha * 0.65 * breathe;
    ctx.strokeStyle = vessel.color;
    ctx.lineWidth = 1.2;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    drawSmoothPath(ctx, pastPoints);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;

    if (futurePoints.length >= 2) {
      if (selected) {
        const finalPoint = futurePoints[futurePoints.length - 1];
        const envelopePulse = 0.04 + 0.03 * Math.sin(time * 0.002);
        const envelope = ctx.createLinearGradient(currentCanvas.x, currentCanvas.y, finalPoint.x, finalPoint.y);
        envelope.addColorStop(0, `rgba(94,174,214,${envelopePulse * 3})`);
        envelope.addColorStop(1, `rgba(94,174,214,${envelopePulse})`);
        ctx.fillStyle = envelope;
        ctx.beginPath();
        ctx.moveTo(currentCanvas.x, currentCanvas.y);
        ctx.lineTo(finalPoint.x + 28, finalPoint.y + 14);
        ctx.lineTo(finalPoint.x - 24, finalPoint.y - 18);
        ctx.closePath();
        ctx.fill();
      }
      ctx.strokeStyle = vessel.color;
      ctx.lineWidth = selected ? 2.4 : 1.3;
      ctx.shadowColor = vessel.color;
      ctx.shadowBlur = selected ? 12 : (isHighRisk ? 4 + 8 * riskPulse : 4);
      ctx.setLineDash([8, 4]);
      ctx.lineDashOffset = -time * 0.04; // marching ants — flowing dash
      ctx.beginPath();
      drawSmoothPath(ctx, futurePoints);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.shadowBlur = 0;

      vessel.altBranches.forEach((branch) => {
        const altPoints = [{ x: current.x, y: current.y }, ...branch].map((point) => normToCanvas(point, vp));
        ctx.globalAlpha = alpha * 0.3 * breathe;
        ctx.strokeStyle = vessel.color;
        ctx.lineWidth = 0.8;
        ctx.setLineDash([3, 8]);
        ctx.beginPath();
        drawSmoothPath(ctx, altPoints);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;
      });
    }

    if (layers.waypoints && selected) {
      vessel.waypoints.forEach((waypoint) => {
        const point = normToCanvas(waypoint, vp);
        ctx.fillStyle = vessel.color;
        ctx.beginPath();
        ctx.arc(point.x, point.y, 3, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "rgba(255,255,255,0.5)";
        ctx.lineWidth = 0.8;
        ctx.stroke();
      });
    }

    ctx.save();
    ctx.translate(currentCanvas.x, currentCanvas.y);
    ctx.rotate((vessel.heading - 90) * (Math.PI / 180));
    const size = selected ? 8 : 5;
    ctx.fillStyle = vessel.color;
    ctx.shadowColor = vessel.color;
    ctx.shadowBlur = isHighRisk ? 6 + 12 * riskPulse : (selected ? 16 : 10);
    ctx.beginPath();
    ctx.moveTo(0, -size);
    ctx.lineTo(size * 0.7, size * 0.6);
    ctx.lineTo(0, size * 0.2);
    ctx.lineTo(-size * 0.7, size * 0.6);
    ctx.closePath();
    ctx.fill();
    ctx.strokeStyle = "rgba(255,255,255,0.6)";
    ctx.lineWidth = 0.8;
    ctx.stroke();
    ctx.restore();

    if (selected) {
      const pulseRadius = 5 + 2.5 * Math.sin(time * 0.004);
      // Outer subtle ring
      ctx.strokeStyle = "rgba(94,174,214,0.25)";
      ctx.lineWidth = 1.2;
      ctx.shadowColor = "transparent";
      ctx.shadowBlur = 0;
      ctx.beginPath();
      ctx.arc(currentCanvas.x, currentCanvas.y, size + pulseRadius + 8, 0, Math.PI * 2);
      ctx.stroke();
      // Inner bright ring
      ctx.strokeStyle = "#5eaed6";
      ctx.lineWidth = 1.8;
      ctx.shadowColor = "#5eaed6";
      ctx.shadowBlur = 8 + 6 * Math.sin(time * 0.004);
      ctx.beginPath();
      ctx.arc(currentCanvas.x, currentCanvas.y, size + pulseRadius, 0, Math.PI * 2);
      ctx.stroke();
      ctx.shadowBlur = 0;

      // Draw Leader Line to Floating Panels (Maritime UI DNA)
      ctx.save();
      ctx.beginPath();
      ctx.strokeStyle = isHighRisk ? "rgba(255, 77, 109, 0.45)" : "rgba(94, 174, 214, 0.4)";
      ctx.lineWidth = 0.8;
      ctx.setLineDash([3, 3]);
      ctx.moveTo(currentCanvas.x, currentCanvas.y);
      
      const onLeft = currentCanvas.x < vp.cw / 2;
      const targetX = onLeft ? 352 : vp.cw - 356;
      const bendX = onLeft ? currentCanvas.x - 20 : currentCanvas.x + 20;
      
      ctx.lineTo(bendX, currentCanvas.y);
      ctx.lineTo(targetX, currentCanvas.y);
      ctx.stroke();
      
      // Draw a tiny target pointer circle at panel edge
      ctx.fillStyle = isHighRisk ? "#FF4D6D" : "#5eaed6";
      ctx.beginPath();
      ctx.arc(targetX, currentCanvas.y, 2.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    // Fading history trail dots
    if (selected || pastPoints.length > 3) {
      for (let h = 0; h < pastPoints.length; h++) {
        const fadeAlpha = 0.08 + 0.12 * (h / Math.max(1, pastPoints.length - 1));
        ctx.fillStyle = vessel.color;
        ctx.globalAlpha = fadeAlpha * breathe;
        ctx.beginPath();
        ctx.arc(pastPoints[h].x, pastPoints[h].y, 1.0, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    }
  });
}

function CanvasMap({ layers, onSelect, selectedId, timeline, vessels }: CanvasMapProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const tooltipRef = useRef<HTMLDivElement | null>(null);
  const centerRef = useRef<Point>({ x: 0.5, y: 0.45 });
  const zoomRef = useRef(2.2);
  const panRef = useRef<{ active: boolean; x: number; y: number }>({
    active: false,
    x: 0,
    y: 0,
  });

  useEffect(() => {
    let frame = 0;
    const animate = (time: number) => {
      const canvas = canvasRef.current;
      if (canvas) {
        drawCanvas(
          canvas,
          vessels,
          selectedId,
          timeline,
          layers,
          centerRef.current,
          zoomRef.current,
          time,
        );
      }
      frame = window.requestAnimationFrame(animate);
    };
    frame = window.requestAnimationFrame(animate);
    return () => window.cancelAnimationFrame(frame);
  }, [layers, selectedId, timeline, vessels]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const tooltip = tooltipRef.current;
    if (!canvas || !tooltip) {
      return undefined;
    }

    const findVessel = (event: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const vp = viewport(canvas, centerRef.current, zoomRef.current);
      const point = canvasToNorm(event.clientX - rect.left, event.clientY - rect.top, vp);
      return vessels.find((vessel) => {
        const current = currentPosition(vessel, timeline);
        return Math.hypot(point.x - current.x, point.y - current.y) < 0.03;
      });
    };

    const onMouseDown = (event: MouseEvent) => {
      panRef.current = { active: true, x: event.clientX, y: event.clientY };
      canvas.style.cursor = "grabbing";
    };
    const onMouseUp = () => {
      panRef.current.active = false;
      canvas.style.cursor = "default";
    };
    const onMouseMove = (event: MouseEvent) => {
      if (panRef.current.active) {
        const vp = viewport(canvas, centerRef.current, zoomRef.current);
        const dx = event.clientX - panRef.current.x;
        const dy = event.clientY - panRef.current.y;
        centerRef.current = {
          x: centerRef.current.x - (dx / vp.cw) * vp.viewWidth,
          y: centerRef.current.y - (dy / vp.ch) * vp.viewHeight,
        };
        panRef.current = { active: true, x: event.clientX, y: event.clientY };
        return;
      }
      const vessel = findVessel(event);
      if (!vessel) {
        tooltip.style.display = "none";
        canvas.style.cursor = "default";
        return;
      }
      const rect = canvas.getBoundingClientRect();
      tooltip.style.display = "block";
      tooltip.style.left = `${event.clientX - rect.left + 15}px`;
      tooltip.style.top = `${event.clientY - rect.top - 25}px`;
      tooltip.textContent = `${vessel.name} | ${vessel.deviceId} | ${vessel.speed.toFixed(1)}kn`;
      canvas.style.cursor = "pointer";
    };
    const onClick = (event: MouseEvent) => {
      const vessel = findVessel(event);
      if (vessel) {
        onSelect(vessel.id);
      }
    };
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const before = canvasToNorm(
        event.clientX - rect.left,
        event.clientY - rect.top,
        viewport(canvas, centerRef.current, zoomRef.current),
      );
      zoomRef.current = Math.max(1.2, Math.min(5.5, zoomRef.current * (event.deltaY > 0 ? 0.9 : 1.1)));
      const after = canvasToNorm(
        event.clientX - rect.left,
        event.clientY - rect.top,
        viewport(canvas, centerRef.current, zoomRef.current),
      );
      centerRef.current = {
        x: centerRef.current.x + before.x - after.x,
        y: centerRef.current.y + before.y - after.y,
      };
    };

    canvas.addEventListener("mousedown", onMouseDown);
    window.addEventListener("mouseup", onMouseUp);
    window.addEventListener("mousemove", onMouseMove);
    canvas.addEventListener("click", onClick);
    canvas.addEventListener("wheel", onWheel, { passive: false });
    return () => {
      canvas.removeEventListener("mousedown", onMouseDown);
      window.removeEventListener("mouseup", onMouseUp);
      window.removeEventListener("mousemove", onMouseMove);
      canvas.removeEventListener("click", onClick);
      canvas.removeEventListener("wheel", onWheel);
    };
  }, [onSelect, timeline, vessels]);

  const zoomBy = (factor: number) => {
    zoomRef.current = Math.max(1.2, Math.min(5.5, zoomRef.current * factor));
  };

  const resetView = () => {
    centerRef.current = { x: 0.5, y: 0.45 };
    zoomRef.current = 2.2;
  };

  const focusSelected = () => {
    const selected = vessels.find((vessel) => vessel.id === selectedId);
    if (!selected) {
      return;
    }
    centerRef.current = currentPosition(selected, timeline);
    zoomRef.current = Math.max(2.4, zoomRef.current);
  };

  return (
    <>
      <canvas id="map-canvas" ref={canvasRef} />
      <div className="tooltip" ref={tooltipRef} />
      <div className="map-toolbar" aria-label="地图控制">
        <button onClick={() => zoomBy(1.16)} type="button" aria-label="放大地图">
          +
        </button>
        <button onClick={() => zoomBy(0.86)} type="button" aria-label="缩小地图">
          -
        </button>
        <button onClick={focusSelected} type="button">
          聚焦
        </button>
        <button onClick={resetView} type="button">
          复位
        </button>
      </div>
    </>
  );
}

function sensorSummary(sensor: SensorReading) {
  return Object.entries(sensor.payload)
    .filter(([key]) => !["signature", "hmac", "hmac_sha256"].includes(key))
    .slice(0, 2)
    .map(([key, value]) => `${key}:${String(value)}`)
    .join(" ");
}

function statusDot(status: string) {
  return status === "online" ? "dot-online" : status === "offline" ? "dot-offline" : "dot-warn";
}

function highestRisk(alerts: LiveDashboardData["alerts"]) {
  return alerts.reduce<RiskLevel>((current, alert) => {
    return riskOrder.indexOf(alert.level) > riskOrder.indexOf(current)
      ? alert.level
      : current;
  }, "none");
}

function MiniChart({ points, tone = "cyan" }: { points: number[]; tone?: "cyan" | "muted" }) {
  const min = Math.min(...points);
  const max = Math.max(...points);
  const span = Math.max(1, max - min);
  const path = points
    .map((point, index) => {
      const x = (index / Math.max(1, points.length - 1)) * 100;
      const y = 36 - ((point - min) / span) * 28;
      return `${index === 0 ? "M" : "L"}${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg className={`mini-chart mini-chart-${tone}`} viewBox="0 0 100 42">
      <path d={path} />
    </svg>
  );
}

function TelemetryRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="data-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function NavGlyph({ icon }: { icon: PanelKey }) {
  switch (icon) {
    case "overview":
      return (
        <svg aria-hidden="true" viewBox="0 0 24 24">
          <rect x="4" y="4" width="7" height="7" rx="1.5" />
          <rect x="13" y="4" width="7" height="7" rx="1.5" />
          <rect x="4" y="13" width="7" height="7" rx="1.5" />
          <path d="M14 16h5" />
          <path d="M14 20h5" />
        </svg>
      );
    case "map":
      return (
        <svg aria-hidden="true" viewBox="0 0 24 24">
          <path d="M4 7l5-2 6 2 5-2v12l-5 2-6-2-5 2z" />
          <path d="M9 5v12" />
          <path d="M15 7v12" />
        </svg>
      );
    case "tracks":
      return (
        <svg aria-hidden="true" viewBox="0 0 24 24">
          <path d="M4 17c4-8 7 2 11-6 1.4-2.8 3.2-3.6 5-3.1" />
          <circle cx="4" cy="17" r="1.6" />
          <circle cx="15" cy="11" r="1.6" />
          <circle cx="20" cy="8" r="1.6" />
        </svg>
      );
    case "prediction":
      return (
        <svg aria-hidden="true" viewBox="0 0 24 24">
          <path d="M4 18l5-5 4 3 7-9" />
          <path d="M15 7h5v5" />
          <path d="M5 7h4" />
          <path d="M5 11h2" />
        </svg>
      );
    case "alerts":
      return (
        <svg aria-hidden="true" viewBox="0 0 24 24">
          <path d="M12 4l9 16H3z" />
          <path d="M12 9v5" />
          <path d="M12 17h.01" />
        </svg>
      );
    case "ai":
      return (
        <svg aria-hidden="true" viewBox="0 0 24 24">
          <rect x="6" y="6" width="12" height="12" rx="2" />
          <path d="M9 3v3" />
          <path d="M15 3v3" />
          <path d="M9 18v3" />
          <path d="M15 18v3" />
          <path d="M3 9h3" />
          <path d="M3 15h3" />
          <path d="M18 9h3" />
          <path d="M18 15h3" />
          <circle cx="12" cy="12" r="2" />
        </svg>
      );
  }
}

export default function Dashboard({
  data,
  realtimeStatus,
}: {
  data: LiveDashboardData;
  realtimeStatus: RealtimeStatus;
}) {
  const now = useClock();
  const [activePanel, setActivePanel] = useState<PanelKey>("overview");
  const [selectedVesselId, setSelectedVesselId] = useState("");
  const [timeline, setTimeline] = useState(0.5);
  const [searchTerm, setSearchTerm] = useState("");
  const [layers, setLayers] = useState<Record<LayerKey, boolean>>({
    heat: true,
    lanes: true,
    radar: true,
    waypoints: true,
  });

  const normalizedSearch = searchTerm.trim().toLowerCase();
  const visibleVessels = useMemo(
    () =>
      normalizedSearch
        ? data.vessels.filter(
            (vessel) =>
              vessel.id.toLowerCase().includes(normalizedSearch) ||
              vessel.deviceId.toLowerCase().includes(normalizedSearch) ||
              vessel.className.toLowerCase().includes(normalizedSearch),
          )
        : data.vessels,
    [data.vessels, normalizedSearch],
  );
  const visibleAlerts = useMemo(
    () =>
      normalizedSearch
        ? data.alerts.filter(
            (alert) =>
              alert.vesselId.toLowerCase().includes(normalizedSearch) ||
              alert.deviceId.toLowerCase().includes(normalizedSearch) ||
              alert.message.toLowerCase().includes(normalizedSearch),
          )
        : data.alerts,
    [data.alerts, normalizedSearch],
  );
  const displayVessels = visibleVessels.length > 0 ? visibleVessels : data.vessels;
  const displayAlerts = visibleAlerts.length > 0 ? visibleAlerts : data.alerts;
  const uiVessels = useMemo(() => toUiVessels(displayVessels), [displayVessels]);
  const selectedVessel =
    displayVessels.find((vessel) => vessel.id === selectedVesselId) ??
    displayVessels[0] ??
    data.vessels[0];
  const selectedUiVessel =
    uiVessels.find((vessel) => vessel.id === selectedVessel?.id) ?? uiVessels[0];
  const selectedAlert =
    displayAlerts.find((alert) => alert.vesselId === selectedVessel?.id) ??
    displayAlerts[0] ??
    data.alerts[0];
  const highRisk = highestRisk(data.alerts);
  const sensorCopy =
    data.sensorReadings.length > 0
      ? sensorSummary(data.sensorReadings[0])
      : "NO LIVE SENSOR DATA";
  const lastUpdatedLabel = data.lastUpdated
    ? data.lastUpdated.toLocaleTimeString("zh-CN", { hour12: false })
    : "等待实时刷新";
  const searchMatchLabel = normalizedSearch
    ? `${visibleVessels.length} 船 / ${visibleAlerts.length} 警`
    : `${data.vessels.length} 船 / ${data.alerts.length} 警`;

  useEffect(() => {
    if (!selectedVesselId && data.vessels[0]) {
      setSelectedVesselId(data.vessels[0].id);
      return;
    }
    if (
      selectedVesselId &&
      !data.vessels.some((vessel) => vessel.id === selectedVesselId) &&
      data.vessels[0]
    ) {
      setSelectedVesselId(data.vessels[0].id);
    }
  }, [data.vessels, selectedVesselId]);

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      )
        return;

      switch (e.key) {
        case "Escape":
          setSearchTerm("");
          break;
        case "ArrowUp":
        case "ArrowDown": {
          e.preventDefault();
          const idx = displayVessels.findIndex(
            (v) => v.id === selectedVesselId,
          );
          const next =
            e.key === "ArrowUp"
              ? Math.max(0, idx - 1)
              : Math.min(displayVessels.length - 1, idx + 1);
          if (displayVessels[next]) {
            setSelectedVesselId(displayVessels[next].id);
          }
          break;
        }
        case "r":
        case "R":
          setLayers((c) => ({ ...c, radar: !c.radar }));
          break;
        case "h":
        case "H":
          setLayers((c) => ({ ...c, heat: !c.heat }));
          break;
        case "l":
        case "L":
          setLayers((c) => ({ ...c, lanes: !c.lanes }));
          break;
        case "w":
        case "W":
          setLayers((c) => ({ ...c, waypoints: !c.waypoints }));
          break;
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [displayVessels, selectedVesselId]);

  const toggleLayer = (layer: LayerKey) => {
    setLayers((current) => ({ ...current, [layer]: !current[layer] }));
  };

  return (
    <div className="ui-console">
      <header className="ui-command-header glass-panel">
        <div className="brand-chip">
          <svg viewBox="0 0 24 24">
            <circle cx="12" cy="12" r="10" />
            <circle cx="12" cy="12" r="4" />
            <circle cx="12" cy="12" r="1.5" />
          </svg>
          <strong>
            海御<span>IoT</span>
          </strong>
        </div>
        <div className="sep-v" />
        <label className="ui-search glass-panel-light">
          <span>搜索</span>
          <input
            aria-label="按轨迹、设备或告警搜索"
            onChange={(event) => setSearchTerm(event.target.value)}
            placeholder="track / device / alert"
            value={searchTerm}
          />
          <small>{searchMatchLabel}</small>
        </label>
        <label className="ui-token glass-panel-light">
          <span>令牌</span>
          <input
            aria-label="API Bearer token"
            onChange={(event) => data.setApiToken(event.target.value)}
            placeholder="Bearer token"
            type="password"
            value={data.apiToken}
          />
        </label>
        <button className="ui-refresh glass-panel-light" onClick={() => void data.refresh()} type="button">
          刷新数据
        </button>
        <div className="sep-v" />
        <div className="ui-oracle">
          <span>API</span>
          <i className={statusDot(data.apiStatus)} />
        </div>
        <div className="sep-v" />
        <time className="ui-clock tabular-nums">UTC {now.toISOString().slice(11, 23)}</time>
        <div className="sep-v" />
        <div className="ui-updated">更新 {lastUpdatedLabel}</div>
        <div className="sep-v" />
        <div className="ui-weather" title={sensorCopy}>{sensorCopy}</div>
        <div className="sep-v" />
        <div className="ui-state">
          <i className={statusDot(data.mode === "offline" ? "offline" : "online")} />
          数据 {data.mode.toUpperCase()}
        </div>
        <div className="sep-v" />
        <div className="ui-links">
          MQTT <i className={statusDot(realtimeStatus.mqtt)} /> WS <i className={statusDot(realtimeStatus.ws)} />
        </div>
        <div className="sep-v" />
        <div className="ui-edge">EDGE {realtimeStatus.edge.toUpperCase()}</div>
        <div className="prototype-note">样机级，未经实港验证</div>
      </header>

      <div className="ui-main">
        <nav className="ui-sidebar glass-panel">
          {navItems.map((item) => (
            <button
              aria-pressed={activePanel === item.key}
              className={activePanel === item.key ? "active" : ""}
              key={item.key}
              onClick={() => setActivePanel(item.key)}
              title={item.label}
              type="button"
            >
              <NavGlyph icon={item.key} />
              <em>{item.label}</em>
            </button>
          ))}
          <small>A1</small>
        </nav>

        <aside className="ui-left">
          <section className="glass-panel info-card">
            <div className="card-title">
              <span>船舶遥测</span>
              <i className="dot-online" />
            </div>
            <div className="sep-h" />
            <TelemetryRow label="ID" value={selectedVessel?.id ?? "-"} />
            <TelemetryRow label="设备" value={selectedVessel?.deviceId ?? "-"} />
            <TelemetryRow label="速度" value={`${selectedVessel?.speedKnots ?? 0} kn`} />
            <TelemetryRow label="航向" value={`${selectedVessel?.courseDeg ?? 0}°`} />
            <TelemetryRow label="CPA" value={`${selectedVessel?.cpaMeters ?? "-"} m`} />
            <TelemetryRow label="TCPA" value={`${selectedVessel?.tcpaSeconds ?? "-"} s`} />
          </section>

          <section className="glass-panel info-card">
            <div className="card-title"><span>传感器流</span></div>
            <div className="sep-h" />
            {data.sensorReadings.length === 0 ? (
              <p className="truth-empty">暂无实时传感器数据，等待 /api/v1/sensors</p>
            ) : (
              data.sensorReadings.slice(0, 4).map((sensor) => (
                <div className="sensor-line" key={sensor.id}>
                  <strong>{sensor.deviceId}</strong>
                  <span>{sensor.sensorType}</span>
                  <em>{sensorSummary(sensor)}</em>
                </div>
              ))
            )}
          </section>

          <section className="glass-panel info-card anomaly-card">
            <div className="card-title"><span>异常标记</span></div>
            <p>
              {selectedAlert
                ? `${selectedAlert.vesselId}: ${selectedAlert.message}`
                : "当前范围内暂无已接入告警。"}
            </p>
          </section>
        </aside>

        <main className="ui-map glass-panel" id="map-container">
          <CanvasMap
            layers={layers}
            onSelect={setSelectedVesselId}
            selectedId={selectedVessel?.id ?? ""}
            timeline={timeline}
            vessels={uiVessels}
          />
          <div className="map-label top-left">EDGE COORDINATE / SCHEMATIC</div>
          <div className="map-label top-right">PAN + WHEEL ZOOM</div>
          <div className="map-label bottom-left">TRAJECTORY PREDICTION · {data.mode.toUpperCase()}</div>
          <div className="map-label bottom-right">SENSORS: {data.sensorReadings.length}</div>
          <div className="compass">N</div>
          <div className="layer-box glass-panel-light">
            {(Object.keys(layers) as LayerKey[]).map((layer) => (
              <label key={layer}>
                <input
                  aria-label={`切换${layerLabels[layer]}`}
                  checked={layers[layer]}
                  onChange={() => toggleLayer(layer)}
                  type="checkbox"
                />
                {layerLabels[layer]}
              </label>
            ))}
          </div>
          <div className="timeline-control">
            <span>历史</span>
            <input
              aria-label="轨迹时间轴"
              max="1"
              min="0"
              onChange={(event) => setTimeline(Number(event.target.value))}
              step="0.01"
              type="range"
              value={timeline}
            />
            <span>预测</span>
          </div>
        </main>

        <aside className="ui-right">
          <section className="glass-panel info-card">
            <div className="card-title">
              <span>CPA / TCPA 分析</span>
              <em>{displayAlerts.length} 组</em>
            </div>
            <div className="sep-h" />
            <div className="risk-list" aria-label="CPA 和 TCPA 风险列表">
              {displayAlerts.map((alert) => (
                <button
                  className={`risk-row risk-${alert.level}${alert.vesselId === selectedVessel?.id ? " selected" : ""}`}
                  key={alert.id}
                  onClick={() => setSelectedVesselId(alert.vesselId)}
                  type="button"
                >
                  <span>{alert.vesselId}</span>
                  <em>CPA:{alert.cpaMeters ?? "-"}m · TCPA:{alert.tcpaSeconds ?? "-"}s</em>
                  <strong>{riskLabel[alert.level]}</strong>
                </button>
              ))}
            </div>
          </section>

          <section className="glass-panel info-card">
            <div className="card-title"><span>威胁排序</span></div>
            <div className="sep-h" />
            {displayAlerts.map((alert, index) => (
              <button
                className={`threat-line${alert.vesselId === selectedVessel?.id ? " selected" : ""}`}
                key={alert.id}
                onClick={() => setSelectedVesselId(alert.vesselId)}
                type="button"
              >
                <span>{index + 1}. {alert.vesselId}</span>
                <strong>{riskLabel[alert.level]}</strong>
              </button>
            ))}
          </section>

          <section className="glass-panel info-card alert-card">
            <div className="card-title">
              <span>告警流</span>
              <i className={displayAlerts.length ? "dot-online" : "dot-warn"} />
            </div>
            <div className="sep-h" />
            {displayAlerts.map((alert) => (
              <p key={alert.id}>
                <time>{alert.time}</time> {alert.message}
              </p>
            ))}
          </section>

          <section className="glass-panel info-card">
            <div className="card-title"><span>到达预测</span></div>
            <div className="sep-h" />
            {displayVessels.slice(0, 3).map((vessel) => (
              <TelemetryRow
                key={vessel.id}
                label={vessel.id}
                value={vessel.tcpaSeconds ? `${vessel.tcpaSeconds}s TCPA` : "暂无实时 TCPA"}
              />
            ))}
          </section>
        </aside>
      </div>

      <footer className="ui-bottom">
        <section className="glass-panel metric-card">
          <span>RK3588 边缘 AI</span>
          <strong>51ms</strong>
          <em>样机级，未经实港验证</em>
        </section>
        <section className="glass-panel metric-card">
          <span>YOLOv8 推理</span>
          <strong>28fps</strong>
          <em>样机级，未经实港验证</em>
        </section>
        <section className="glass-panel metric-card">
          <span>MQTT / 网络</span>
          <strong>{realtimeStatus.mqtt.toUpperCase()}</strong>
          <em>接口接线，非吞吐实测</em>
        </section>
        <section className="glass-panel chart-card">
          <span>告警趋势</span>
          <MiniChart points={[12, 15, 13, 17, 14, 16, 15, 18, 16, 17]} />
          <em>近 10 个采样窗口的告警量变化</em>
        </section>
        <section className="glass-panel chart-card">
          <span>轨迹吞吐</span>
          <MiniChart points={[90, 110, 96, 124, 118, 131, 122, 140, 135, 146]} tone="muted" />
          <em>边缘上报轨迹事件数量</em>
        </section>
        <section className="glass-panel metric-card compact">
          <span>风险</span>
          <strong>{riskLabel[highRisk]}</strong>
          <em>{data.alerts.length} 条告警</em>
        </section>
        <section className="glass-panel metric-card compact">
          <span>置信度</span>
          <strong>{Math.round((selectedUiVessel?.source.confidence ?? 0) * 100)}%</strong>
          <em>当前目标</em>
        </section>
      </footer>
    </div>
  );
}

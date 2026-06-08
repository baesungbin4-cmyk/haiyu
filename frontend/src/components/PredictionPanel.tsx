import { useEffect, useRef } from "react";

import {
  riskLabel,
  type VesselTrack,
} from "../services/mockData";
import { echarts } from "./chartSetup";

interface PredictionPanelProps {
  vessel: VesselTrack;
}

export default function PredictionPanel({ vessel }: PredictionPanelProps) {
  const chartRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!chartRef.current) {
      return undefined;
    }
    const chart = echarts.init(chartRef.current);
    const points = [vessel.position, ...vessel.prediction];
    chart.setOption({
      animation: false,
      grid: { left: 24, right: 10, top: 10, bottom: 18 },
      xAxis: {
        type: "category",
        data: points.map((_, index) => `T+${index * 15}s`),
        axisLine: { lineStyle: { color: "#29435f" } },
        axisLabel: { color: "#8aa3bf", fontSize: 10 },
      },
      yAxis: {
        type: "value",
        axisLine: { lineStyle: { color: "#29435f" } },
        splitLine: { lineStyle: { color: "#1f2d45" } },
        axisLabel: { color: "#8aa3bf", fontSize: 10 },
      },
      series: [
        {
          name: "CPA 外推",
          type: "line",
          smooth: true,
          symbolSize: 5,
          data: points.map((point) => Math.round(100 - point.y)),
          lineStyle: { color: "#2dd4bf", width: 2, type: "dashed" },
          itemStyle: { color: "#22d3ee" },
          areaStyle: { color: "rgba(45, 212, 191, 0.12)" },
        },
      ],
      tooltip: { trigger: "axis" },
    });
    const resize = () => chart.resize();
    const observer = new ResizeObserver(resize);
    observer.observe(chartRef.current);
    return () => {
      observer.disconnect();
      chart.dispose();
    };
  }, [vessel]);

  return (
    <section className="panel prediction-panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">Prediction</span>
          <h2>轨迹预测</h2>
        </div>
      </div>
      <div className="prediction-stats">
        <div>
          <small>目标</small>
          <strong>{vessel.id}</strong>
        </div>
        <div>
          <small>风险</small>
          <strong className={`text-risk-${vessel.riskLevel}`}>
            {riskLabel[vessel.riskLevel]}
          </strong>
        </div>
        <div>
          <small>TCPA</small>
          <strong>{vessel.tcpaSeconds ?? "-"}s</strong>
        </div>
        <div>
          <small>CPA</small>
          <strong>{vessel.cpaMeters ?? "-"}m</strong>
        </div>
      </div>
      <div className="chart prediction-chart" ref={chartRef} />
    </section>
  );
}

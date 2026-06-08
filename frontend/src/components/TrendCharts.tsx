import { useEffect, useRef } from "react";

import { type TrendPoint } from "../services/mockData";
import { echarts } from "./chartSetup";

interface TrendChartsProps {
  points: TrendPoint[];
}

export default function TrendCharts({ points }: TrendChartsProps) {
  const chartRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!chartRef.current) {
      return undefined;
    }
    const chart = echarts.init(chartRef.current);
    chart.setOption({
      animation: false,
      color: ["#22d3ee", "#ff4d6d", "#2dd4bf"],
      grid: { left: 30, right: 32, top: 24, bottom: 20 },
      legend: {
        top: 0,
        right: 0,
        itemHeight: 8,
        itemWidth: 10,
        textStyle: { color: "#8aa3bf", fontSize: 10 },
      },
      tooltip: { trigger: "axis" },
      xAxis: {
        type: "category",
        data: points.map((point) => point.time),
        axisLine: { lineStyle: { color: "#29435f" } },
        axisLabel: { color: "#8aa3bf", fontSize: 10 },
      },
      yAxis: [
        {
          type: "value",
          name: "告警",
          nameTextStyle: { color: "#8aa3bf" },
          splitLine: { lineStyle: { color: "#1f2d45" } },
          axisLabel: { color: "#8aa3bf", fontSize: 10 },
        },
        {
          type: "value",
          name: "延迟",
          nameTextStyle: { color: "#8aa3bf" },
          splitLine: { show: false },
          axisLabel: { color: "#8aa3bf", fontSize: 10 },
        },
      ],
      series: [
        {
          name: "告警数",
          type: "bar",
          barWidth: 12,
          data: points.map((point) => point.alerts),
          itemStyle: { borderRadius: [4, 4, 0, 0] },
        },
        {
          name: "高风险",
          type: "line",
          smooth: true,
          data: points.map((point) => point.highRisk),
          lineStyle: { width: 3 },
        },
        {
          name: "平均延迟",
          type: "line",
          yAxisIndex: 1,
          smooth: true,
          data: points.map((point) => point.avgLatencyMs),
          lineStyle: { width: 2, type: "dashed" },
        },
      ],
    });
    const resize = () => chart.resize();
    const observer = new ResizeObserver(resize);
    observer.observe(chartRef.current);
    return () => {
      observer.disconnect();
      chart.dispose();
    };
  }, [points]);

  return (
    <section className="panel trend-panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">Trend</span>
          <h2>告警与链路趋势</h2>
        </div>
      </div>
      <div className="chart trend-chart" ref={chartRef} />
    </section>
  );
}

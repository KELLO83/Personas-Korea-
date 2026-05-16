"use client";

import { useEffect, useRef } from "react";
import * as echarts from "echarts";

export interface ChartDatum {
  label: string;
  value: number;
}

interface ChartCardProps {
  title: string;
  items: ChartDatum[];
  kind?: "bar" | "pie";
  valueLabel?: string;
}

export function ChartCard({ title, items, kind = "bar", valueLabel = "count" }: ChartCardProps) {
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!ref.current || items.length === 0) return;
    const chart = echarts.init(ref.current);
    const option =
      kind === "pie"
        ? {
            tooltip: { trigger: "item" },
            series: [
              {
                name: valueLabel,
                type: "pie",
                radius: ["42%", "72%"],
                avoidLabelOverlap: true,
                data: items.map((item) => ({ name: item.label, value: item.value })),
              },
            ],
          }
        : {
            tooltip: { trigger: "axis" },
            grid: { left: 8, right: 12, top: 18, bottom: 8, containLabel: true },
            xAxis: { type: "value", axisLabel: { color: "#8a94a6" } },
            yAxis: {
              type: "category",
              data: items.map((item) => item.label),
              axisLabel: { color: "#8a94a6", width: 92, overflow: "truncate" },
            },
            series: [
              {
                name: valueLabel,
                type: "bar",
                data: items.map((item) => item.value),
                itemStyle: { color: "#45b7ff", borderRadius: [0, 6, 6, 0] },
              },
            ],
          };

    chart.setOption(option);
    const resize = () => chart.resize();
    window.addEventListener("resize", resize);
    return () => {
      window.removeEventListener("resize", resize);
      chart.dispose();
    };
  }, [items, kind, valueLabel]);

  return (
    <div className="card chart-card">
      <h3>{title}</h3>
      {items.length === 0 ? <p className="muted small">표시할 데이터가 없습니다.</p> : <div className="chart-canvas" ref={ref} />}
    </div>
  );
}

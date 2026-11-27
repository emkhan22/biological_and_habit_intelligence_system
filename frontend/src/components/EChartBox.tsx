import React, { useEffect, useRef } from "react";
import * as echarts from "echarts";
import type { EChartsOption } from "echarts";

interface Props {
  option: EChartsOption;
  height?: number;
  emptyText?: string;
  isEmpty?: boolean;
}

export const EChartBox: React.FC<Props> = ({
  option,
  height = 240,
  emptyText = "Collecting validated data — missing days remain NaN.",
  isEmpty = false,
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<echarts.ECharts | null>(null);

  useEffect(() => {
    if (isEmpty || !containerRef.current) return;
    // Skip ECharts canvas mount inside jsdom unit tests where clientWidth is 0
    if (typeof window !== "undefined" && containerRef.current.clientWidth === 0) {
      return;
    }

    const instance = echarts.init(containerRef.current, undefined, { renderer: "svg" });
    chartRef.current = instance;
    instance.setOption({
      backgroundColor: "transparent",
      textStyle: { fontFamily: "Inter, system-ui, sans-serif", color: "#94A3B8" },
      ...option,
    });

    const ro =
      typeof ResizeObserver !== "undefined"
        ? new ResizeObserver(() => {
            instance.resize();
          })
        : null;
    ro?.observe(containerRef.current);

    return () => {
      ro?.disconnect();
      instance.dispose();
      chartRef.current = null;
    };
  }, [option, isEmpty]);

  if (isEmpty) {
    return (
      <div
        style={{ height }}
        className="flex items-center justify-center rounded-lg border border-dashed border-[#283042] bg-[#0B0D12]/60 text-xs text-slate-500 px-4 text-center"
      >
        {emptyText}
      </div>
    );
  }

  return <div ref={containerRef} style={{ height, width: "100%" }} />;
};

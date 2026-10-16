import {
  ResponsiveContainer, LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
} from "recharts";

const COLORS = ["#1B3A8C", "#17A2A0", "#D98324"];

export default function Chart({ chart }) {
  const rows = chart.x.map((x, i) => {
    const row = { x };
    chart.series.forEach((s) => { row[s.name] = s.values[i]; });
    return row;
  });
  const isBar = chart.type === "bar";
  const Plot = isBar ? BarChart : LineChart;
  return (
    <figure className="chart">
      <figcaption>{chart.title}</figcaption>
      <ResponsiveContainer width="100%" height={260}>
        <Plot data={rows} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="#dde2ee" vertical={false} />
          <XAxis dataKey="x" tick={{ fontSize: 12 }} minTickGap={24} />
          <YAxis tick={{ fontSize: 12 }} width={48} />
          <Tooltip formatter={(v) => (v == null ? "-" : `${v} ${chart.unit}`)} />
          {chart.series.length > 1 && <Legend />}
          {chart.series.map((s, i) =>
            isBar ? (
              <Bar key={s.name} dataKey={s.name} fill={COLORS[i % 3]} radius={[2, 2, 0, 0]} />
            ) : (
              <Line key={s.name} dataKey={s.name} type="monotone" stroke={COLORS[i % 3]} strokeWidth={2}
                    strokeDasharray={i === 1 ? "6 4" : undefined} dot={false} connectNulls={false} />
            )
          )}
        </Plot>
      </ResponsiveContainer>
    </figure>
  );
}

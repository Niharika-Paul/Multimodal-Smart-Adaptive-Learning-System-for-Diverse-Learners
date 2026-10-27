import { motion } from "framer-motion";
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

const data = [
  { time: "9:00", engagement: 72 },
  { time: "9:10", engagement: 78 },
  { time: "9:20", engagement: 85 },
  { time: "9:30", engagement: 82 },
  { time: "9:40", engagement: 90 },
  { time: "9:50", engagement: 88 },
  { time: "10:00", engagement: 92 },
  { time: "10:10", engagement: 87 },
  { time: "10:20", engagement: 91 },
  { time: "10:30", engagement: 89 },
];

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-card/90 backdrop-blur-md rounded-xl px-4 py-2 shadow-soft">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="text-sm font-semibold text-foreground tabular-nums">
        {payload[0].value}%
      </p>
    </div>
  );
};

const EngagementChart = () => (
  <motion.div
    initial={{ opacity: 0, y: 12 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ delay: 0.4 }}
    className="card-soft"
  >
    <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-4">
      Engagement Over Time
    </h2>
    <div className="h-48 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
          <defs>
            <linearGradient id="engGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="hsl(210, 100%, 82%)" stopOpacity={0.4} />
              <stop offset="100%" stopColor="hsl(210, 100%, 82%)" stopOpacity={0} />
            </linearGradient>
          </defs>
          <XAxis
            dataKey="time"
            axisLine={false}
            tickLine={false}
            tick={{ fontSize: 11, fill: "hsl(215, 16%, 47%)" }}
          />
          <YAxis
            domain={[60, 100]}
            axisLine={false}
            tickLine={false}
            tick={{ fontSize: 11, fill: "hsl(215, 16%, 47%)" }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Area
            type="monotone"
            dataKey="engagement"
            stroke="hsl(210, 100%, 82%)"
            strokeWidth={2.5}
            fill="url(#engGradient)"
            strokeLinecap="round"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  </motion.div>
);

export default EngagementChart;

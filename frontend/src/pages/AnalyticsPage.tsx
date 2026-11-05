import { motion } from "framer-motion";
import TopNav from "@/components/dashboard/TopNav";
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, BarChart, Bar } from "recharts";

const weeklyData = [
  { day: "Mon", engagement: 78, focus: 65 },
  { day: "Tue", engagement: 82, focus: 72 },
  { day: "Wed", engagement: 75, focus: 68 },
  { day: "Thu", engagement: 88, focus: 80 },
  { day: "Fri", engagement: 92, focus: 85 },
  { day: "Sat", engagement: 70, focus: 60 },
  { day: "Sun", engagement: 85, focus: 78 },
];

const sessionData = [
  { session: "S1", duration: 45 },
  { session: "S2", duration: 32 },
  { session: "S3", duration: 58 },
  { session: "S4", duration: 41 },
  { session: "S5", duration: 52 },
  { session: "S6", duration: 38 },
];

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-card/90 backdrop-blur-md rounded-xl px-4 py-2 shadow-soft">
      <p className="text-xs text-muted-foreground">{label}</p>
      {payload.map((p: any, i: number) => (
        <p key={i} className="text-sm font-semibold text-foreground tabular-nums">
          {p.name}: {p.value}{p.name === "duration" ? " min" : "%"}
        </p>
      ))}
    </div>
  );
};

const stats = [
  { label: "Total Study Hours", value: "127h", change: "+12%" },
  { label: "Avg Engagement", value: "84%", change: "+5%" },
  { label: "Sessions This Week", value: "18", change: "+3" },
  { label: "Streak", value: "12 days", change: "🔥" },
];

const AnalyticsPage = () => (
  <div className="min-h-screen bg-background">
    <TopNav />
    <motion.main
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="max-w-7xl mx-auto px-4 sm:px-8 py-8"
    >
      <div className="mb-8">
        <h2 className="text-2xl font-semibold text-foreground">Analytics Overview 📊</h2>
        <p className="text-sm text-muted-foreground mt-1 leading-relaxed">
          Track your learning patterns and find insights to improve.
        </p>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        {stats.map((s) => (
          <motion.div
            key={s.label}
            whileHover={{ y: -3 }}
            className="card-soft text-center"
          >
            <h3 className="text-2xl font-bold text-foreground tabular-nums">{s.value}</h3>
            <p className="text-xs text-muted-foreground mt-1">{s.label}</p>
            <span className="text-xs font-medium text-pastel-sage mt-1 inline-block">{s.change}</span>
          </motion.div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Weekly Engagement */}
        <div className="card-soft">
          <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-4">
            Weekly Engagement & Focus
          </h3>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={weeklyData} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="engGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="hsl(210, 100%, 82%)" stopOpacity={0.4} />
                    <stop offset="100%" stopColor="hsl(210, 100%, 82%)" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="focGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="hsl(335, 100%, 89%)" stopOpacity={0.4} />
                    <stop offset="100%" stopColor="hsl(335, 100%, 89%)" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="day" axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: "hsl(215,16%,47%)" }} />
                <YAxis domain={[50, 100]} axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: "hsl(215,16%,47%)" }} />
                <Tooltip content={<CustomTooltip />} />
                <Area type="monotone" dataKey="engagement" stroke="hsl(210,100%,82%)" strokeWidth={2.5} fill="url(#engGrad)" strokeLinecap="round" />
                <Area type="monotone" dataKey="focus" stroke="hsl(335,100%,89%)" strokeWidth={2.5} fill="url(#focGrad)" strokeLinecap="round" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Session Durations */}
        <div className="card-soft">
          <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-4">
            Session Durations
          </h3>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={sessionData} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
                <XAxis dataKey="session" axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: "hsl(215,16%,47%)" }} />
                <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: "hsl(215,16%,47%)" }} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="duration" fill="hsl(210,100%,82%)" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </motion.main>
  </div>
);

export default AnalyticsPage;

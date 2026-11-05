import { motion } from "framer-motion";
import { Timer, AlertCircle, TrendingUp } from "lucide-react";

const stats = [
  { label: "Focus Time", value: "42 min", icon: Timer, color: "text-pastel-blue" },
  { label: "Distraction Count", value: "3", icon: AlertCircle, color: "text-pastel-peach" },
  { label: "Avg Engagement", value: "86%", icon: TrendingUp, color: "text-pastel-sage" },
];

const SessionSummary = () => (
  <motion.div
    initial={{ opacity: 0, y: 12 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ delay: 0.5 }}
    className="grid grid-cols-3 gap-4"
  >
    {stats.map((s) => {
      const Icon = s.icon;
      return (
        <div key={s.label} className="card-soft flex flex-col items-center gap-2 text-center">
          <Icon size={20} className={s.color} />
          <h3 className="text-xl font-bold text-foreground tabular-nums">{s.value}</h3>
          <span className="text-xs text-muted-foreground">{s.label}</span>
        </div>
      );
    })}
  </motion.div>
);

export default SessionSummary;

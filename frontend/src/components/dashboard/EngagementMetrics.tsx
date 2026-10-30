import { motion } from "framer-motion";
import { Activity, Eye, Zap, Brain, ScanEye } from "lucide-react";
import type { EyeMetrics } from "@/components/dashboard/WebcamPanel";

interface MetricProps {
  label: string;
  value: string;
  change?: string;
  detail?: string;
  icon: React.ElementType;
  borderColor: string;
  progress?: number;
}

const CircularProgress = ({
  value,
  color,
}: {
  value: number;
  color: string;
}) => {
  const circumference = 2 * Math.PI * 28;
  const offset = circumference - (value / 100) * circumference;

  return (
    <svg width="68" height="68" className="transform -rotate-90">
      <circle
        cx="34"
        cy="34"
        r="28"
        fill="none"
        stroke="hsl(var(--muted))"
        strokeWidth="5"
      />
      <circle
        cx="34"
        cy="34"
        r="28"
        fill="none"
        stroke={color}
        strokeWidth="5"
        strokeLinecap="round"
        strokeDasharray={circumference}
        strokeDashoffset={offset}
        className="transition-all duration-700 ease-out"
      />
    </svg>
  );
};

const MetricCard = ({
  label,
  value,
  change,
  detail,
  icon: Icon,
  borderColor,
  progress,
}: MetricProps) => (
  <motion.div
    whileHover={{ y: -5, transition: { duration: 0.2 } }}
    className={`bg-card p-5 rounded-[2rem] shadow-soft border-t-4 ${borderColor} flex flex-col gap-3`}
  >
    <div className="flex items-center justify-between gap-2">
      <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
        {label}
      </span>
      <Icon size={16} className="text-muted-foreground shrink-0" />
    </div>
    <div className="flex items-center gap-3">
      {progress !== undefined && (
        <CircularProgress
          value={progress}
          color="hsl(var(--pastel-blue))"
        />
      )}
      <div>
        <h3 className="text-2xl font-bold text-foreground tabular-nums">
          {value}
        </h3>
        {change && (
          <span className="text-xs font-medium text-muted-foreground">
            {change}
          </span>
        )}
        {detail && (
          <span className="block text-xs text-muted-foreground">
            {detail}
          </span>
        )}
      </div>
    </div>
  </motion.div>
);

interface EngagementMetricsProps {
  metrics: EyeMetrics | null;
}

const EngagementMetrics = ({ metrics }: EngagementMetricsProps) => {
  const cards: MetricProps[] = [
    {
      label: "Model Confidence",
      value: metrics?.engagement_confidence == null
        ? metrics?.engagement_status === "Collecting clip frames"
          ? `Collecting ${metrics.engagement_frames_collected ?? 0}/${metrics.engagement_frames_required ?? 65}`
          : metrics?.engagement_status === "Extracting I3D features"
            ? "Extracting I3D features…"
            : metrics?.engagement_status === "Inference error"
            ? "Inference unavailable"
            : "Waiting for prediction"
        : `${(metrics.engagement_confidence * 100).toFixed(1)}%`,
      change: metrics?.engagement_checkpoint
        ? `W0-P · ${metrics.engagement_personalization_group}`
        : undefined,
      icon: Activity,
      borderColor: "border-pastel-blue",
      progress: metrics?.engagement_confidence == null
        ? undefined
        : metrics.engagement_confidence * 100,
    },
    {
      label: "Estimated Eye Contact",
      value: metrics?.eye_contact_percent == null
        ? metrics?.face_detected ? "Measuring gaze…" : "Waiting for face"
        : `${metrics.eye_contact_percent.toFixed(1)}%`,
      change: metrics?.gaze_status ?? "Iris position within both eye openings",
      icon: Eye,
      borderColor: "border-pastel-pink",
    },
    {
      label: "Blink Rate",
      value: metrics?.face_detected && metrics.blink_rate_ready
        ? metrics.blink_rate == null
          ? "Rate unavailable"
          : `${metrics.blink_rate.toFixed(1)} blinks/min`
        : metrics?.face_detected
          ? `${metrics.total_blinks} blinks`
          : "Waiting for face",
      change: metrics?.face_detected && metrics.blink_rate_ready
        ? "Calculated from completed 60-second window"
        : metrics?.face_detected
          ? "Current blink count"
          : "Observation starts when a face is detected",
      detail: metrics?.face_detected && metrics.blink_rate_ready
        ? `${metrics.blink_window_count ?? 0} blinks in the rolling 60 seconds · ${metrics.total_blinks} session total`
        : metrics?.face_detected
          ? (() => {
              const elapsed = Math.min(60, Math.floor(metrics.blink_rate_elapsed_seconds ?? 0));
              return `${elapsed} seconds elapsed · ${60 - elapsed} seconds remaining · 60 seconds total`;
            })()
          : undefined,
      icon: Zap,
      borderColor: "border-pastel-peach",
    },
    {
      label: "Normalized Iris Ratio",
      value: !metrics
        ? "Waiting…"
        : !metrics.face_detected
          ? "No face detected"
          : !metrics.baseline_ready
            ? "Calibrating…"
            : metrics.norm_pupil == null
              ? "Unavailable"
              : `${metrics.norm_pupil.toFixed(2)}×`,
      icon: ScanEye,
      borderColor: "border-pastel-lavender",
    },
    {
      label: "Attention Level",
      value: metrics?.engagement_label ?? (metrics?.engagement_status === "Inference error"
        ? "Unavailable"
        : metrics?.engagement_status === "Waiting for face"
          ? "Waiting for face"
          : "Collecting video"),
      change: metrics?.engagement_error,
      icon: Brain,
      borderColor: "border-pastel-sage",
    },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.2 }}
      className="flex flex-col gap-4"
    >
      <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">
        Live Engagement & Eye Metrics
      </h2>
      <div className="grid grid-cols-2 gap-4">
        {cards.slice(0, 4).map((card) => (
          <MetricCard key={card.label} {...card} />
        ))}
      </div>
      <MetricCard {...cards[4]} />
    </motion.div>
  );
};

export default EngagementMetrics;

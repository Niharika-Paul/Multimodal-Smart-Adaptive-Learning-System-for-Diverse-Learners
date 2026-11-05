import { motion } from "framer-motion";
import { useState } from "react";

interface ModuleToggle {
  label: string;
  enabled: boolean;
  available: boolean;
}

const initialModules: ModuleToggle[] = [
  { label: "Eye Tracking", enabled: true, available: true },
  { label: "Engagement Detection", enabled: true, available: true },
  { label: "Audio Analysis", enabled: false, available: false },
  { label: "Handwriting Analysis", enabled: false, available: false },
];

const Toggle = ({
  enabled,
  available,
  onToggle,
}: {
  enabled: boolean;
  available: boolean;
  onToggle: () => void;
}) => (
  <button
    onClick={available ? onToggle : undefined}
    className={`relative w-11 h-6 rounded-full transition-colors duration-200 ${
      !available
        ? "bg-muted cursor-not-allowed opacity-50"
        : enabled
        ? "bg-pastel-blue"
        : "bg-muted"
    }`}
  >
    <motion.div
      layout
      className="absolute top-0.5 w-5 h-5 rounded-full bg-card shadow-soft"
      style={{ left: enabled && available ? "calc(100% - 22px)" : "2px" }}
      transition={{ type: "spring", stiffness: 500, damping: 30 }}
    />
  </button>
);

const MultimodalModules = () => {
  const [modules, setModules] = useState(initialModules);

  const toggle = (index: number) => {
    setModules((prev) =>
      prev.map((m, i) => (i === index ? { ...m, enabled: !m.enabled } : m))
    );
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.6 }}
      className="card-soft"
    >
      <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-4">
        Multimodal Modules
      </h2>
      <div className="flex flex-col gap-3">
        {modules.map((mod, i) => (
          <div key={mod.label} className="flex items-center justify-between">
            <span
              className={`text-sm ${
                mod.available ? "text-foreground" : "text-muted-foreground"
              }`}
            >
              {mod.label}
              {!mod.available && (
                <span className="ml-2 text-[10px] uppercase tracking-wider text-muted-foreground/60">
                  Coming soon
                </span>
              )}
            </span>
            <Toggle
              enabled={mod.enabled}
              available={mod.available}
              onToggle={() => toggle(i)}
            />
          </div>
        ))}
      </div>
    </motion.div>
  );
};

export default MultimodalModules;

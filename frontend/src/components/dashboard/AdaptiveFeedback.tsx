import { motion } from "framer-motion";
import { Sparkles, Coffee, SlidersHorizontal } from "lucide-react";

const messages = [
  {
    text: "You seem slightly distracted — that's okay! It happens to everyone.",
    icon: "💭",
  },
  {
    text: "Try switching to interactive mode for a fresh approach ✨",
    icon: "🎯",
  },
  {
    text: "You've been focused for 42 minutes — that's 15% longer than yesterday!",
    icon: "🌟",
  },
];

const AdaptiveFeedback = () => (
  <motion.div
    initial={{ opacity: 0, y: 12 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ delay: 0.3 }}
    className="bg-secondary/30 p-6 rounded-[2rem] border border-secondary flex flex-col gap-5"
  >
    <div className="flex items-center gap-2">
      <Sparkles size={18} className="text-pastel-pink" />
      <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">
        Your Helper
      </h2>
    </div>

    <div className="flex flex-col gap-3">
      {messages.map((msg, i) => (
        <motion.div
          key={i}
          initial={{ opacity: 0, x: 8 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.4 + i * 0.1 }}
          className="flex gap-3 items-start"
        >
          <span className="text-lg mt-0.5">{msg.icon}</span>
          <p className="text-sm leading-relaxed text-foreground/80">{msg.text}</p>
        </motion.div>
      ))}
    </div>

    <div className="flex flex-col gap-2 mt-1">
      <button className="flex items-center justify-center gap-2 bg-card text-foreground shadow-soft hover:shadow-hover-effect transition-all rounded-2xl py-3 px-4 text-sm font-medium">
        <SlidersHorizontal size={14} />
        Adjust Content
      </button>
      <button className="flex items-center justify-center gap-2 bg-transparent border border-secondary text-pastel-rose hover:bg-secondary/30 transition-colors rounded-2xl py-3 px-4 text-sm font-medium">
        <Coffee size={14} />
        Take a short break
      </button>
    </div>
  </motion.div>
);

export default AdaptiveFeedback;

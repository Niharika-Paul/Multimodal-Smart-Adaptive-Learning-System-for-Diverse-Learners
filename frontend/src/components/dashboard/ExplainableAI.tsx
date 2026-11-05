import { motion } from "framer-motion";
import { Lightbulb } from "lucide-react";

const ExplainableAI = () => (
  <motion.div
    initial={{ opacity: 0, y: 12 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ delay: 0.55 }}
    className="bg-pastel-peach/20 p-5 rounded-[2rem] border border-pastel-peach/30"
  >
    <div className="flex items-start gap-3">
      <div className="mt-0.5 w-8 h-8 rounded-xl bg-pastel-peach/40 flex items-center justify-center flex-shrink-0">
        <Lightbulb size={16} className="text-foreground/70" />
      </div>
      <div>
        <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-1">
          AI Insight
        </h3>
        <p className="text-sm leading-relaxed text-foreground/80">
          Lower engagement noticed due to reduced eye fixation and increased blink rate. A short interactive activity might help!
        </p>
      </div>
    </div>
  </motion.div>
);

export default ExplainableAI;

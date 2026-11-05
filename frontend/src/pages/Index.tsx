import TopNav from "@/components/dashboard/TopNav";
import { useCallback, useState } from "react";
import WebcamPanel, { type EyeMetrics } from "@/components/dashboard/WebcamPanel";
import EngagementMetrics from "@/components/dashboard/EngagementMetrics";
import AdaptiveFeedback from "@/components/dashboard/AdaptiveFeedback";
import EngagementChart from "@/components/dashboard/EngagementChart";
import SessionSummary from "@/components/dashboard/SessionSummary";
import ExplainableAI from "@/components/dashboard/ExplainableAI";
import MultimodalModules from "@/components/dashboard/MultimodalModules";

const Index = () => {
  const [eyeMetrics, setEyeMetrics] = useState<EyeMetrics | null>(null);

  const handleMetrics = useCallback((metrics: EyeMetrics) => {
    setEyeMetrics(metrics);
  }, []);
  return (
    <div className="min-h-screen bg-background">
      <TopNav />

      <main className="max-w-7xl mx-auto px-4 sm:px-8 py-8">
        {/* Welcome */}
        <div className="mb-8">
          <h2 className="text-2xl font-semibold text-foreground">
            Welcome back, Alex 👋
          </h2>
          <p className="text-sm text-muted-foreground mt-1 leading-relaxed">
            Let's find your flow today. Here's how your session is going.
          </p>
        </div>

        {/* Main 3-column grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left — Webcam */}
          <div className="lg:col-span-3">
            <WebcamPanel onMetrics={handleMetrics} />
          </div>

          {/* Center — Metrics */}
          <div className="lg:col-span-5">
            <EngagementMetrics metrics={eyeMetrics} />
          </div>

          {/* Right — Feedback */}
          <div className="lg:col-span-4">
            <AdaptiveFeedback />
          </div>
        </div>

        {/* Lower section */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 mt-6">
          <div className="lg:col-span-8 flex flex-col gap-6">
            <EngagementChart />
            <SessionSummary />
          </div>

          <div className="lg:col-span-4 flex flex-col gap-6">
            <ExplainableAI />
            <MultimodalModules />
          </div>
        </div>
      </main>
    </div>
  );
};

export default Index;

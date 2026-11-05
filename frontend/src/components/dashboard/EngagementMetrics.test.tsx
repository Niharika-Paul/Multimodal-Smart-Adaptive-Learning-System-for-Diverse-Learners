import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import EngagementMetrics from "@/components/dashboard/EngagementMetrics";
import type { EyeMetrics } from "@/components/dashboard/WebcamPanel";

const metrics = (overrides: Partial<EyeMetrics> = {}): EyeMetrics => ({
  face_detected: true,
  status: "Tracking",
  total_blinks: 214,
  blink_rate: null,
  iris_eye_ratio: 0.2,
  norm_pupil: 1,
  baseline_ready: true,
  engagement_score: null,
  blink_rate_ready: false,
  blink_window_count: 7,
  blink_rate_elapsed_seconds: 17,
  eye_contact_percent: 50,
  engagement_status: "Collecting clip frames",
  engagement_label: null,
  engagement_confidence: null,
  ...overrides,
});

describe("Blink Rate card", () => {
  it("shows only Calculating while the new observation window is incomplete", () => {
    render(<EngagementMetrics metrics={metrics()} />);

    expect(screen.getByText("Calculating...")).toBeInTheDocument();
    expect(screen.queryByText(/214 blinks|Current blink count|seconds elapsed|seconds remaining|session total/i)).not.toBeInTheDocument();
  });

  it("shows only the calculated rate and clears stale values when metrics reset", () => {
    const { rerender } = render(
      <EngagementMetrics metrics={metrics({ blink_rate: 17, blink_rate_ready: true })} />,
    );

    expect(screen.getByText("17 blinks/min")).toBeInTheDocument();
    expect(screen.queryByText(/214 blinks|Current blink count|seconds elapsed|seconds remaining|session total/i)).not.toBeInTheDocument();

    rerender(<EngagementMetrics metrics={null} />);
    expect(screen.queryByText("17 blinks/min")).not.toBeInTheDocument();
    expect(screen.getByText("Calculating...")).toBeInTheDocument();
  });
});

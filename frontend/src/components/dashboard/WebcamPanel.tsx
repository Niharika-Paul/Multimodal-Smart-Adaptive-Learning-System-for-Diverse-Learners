import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Camera, Eye, ScanFace, Square, Circle } from "lucide-react";

export interface EyeMetrics {
  face_detected: boolean;
  status: string;
  total_blinks: number;
  blink_rate: number | null;
  iris_eye_ratio: number | null;
  norm_pupil: number | null;
  baseline_ready: boolean;
  engagement_score: number | null;
  blink_rate_ready: boolean;
}

interface WebcamPanelProps {
  onMetrics: (metrics: EyeMetrics) => void;
}

const API_URL = "http://127.0.0.1:8000";

const WebcamPanel = ({ onMetrics }: WebcamPanelProps) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const busyRef = useRef(false);
  const [cameraOn, setCameraOn] = useState(false);
  const [status, setStatus] = useState("Camera is off");
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    let intervalId: number | undefined;

    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
          audio: false,
        });

        if (!active) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }

        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }

        setCameraOn(true);
        setStatus("Connecting to eye tracker");

        intervalId = window.setInterval(async () => {
          const video = videoRef.current;
          const canvas = canvasRef.current;

          if (!video || !canvas || video.readyState < 2 || busyRef.current) return;

          busyRef.current = true;
          try {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const context = canvas.getContext("2d");
            if (!context) return;

            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            const image = canvas.toDataURL("image/jpeg", 0.75);

            const response = await fetch(`${API_URL}/analyze-frame`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ image }),
            });

            if (!response.ok) throw new Error(`Eye tracker returned ${response.status}`);

            const data: EyeMetrics = await response.json();
            if (active) {
              onMetrics(data);
              setStatus(data.status);
              setError("");
            }
          } catch (err) {
            console.error("Eye tracker request failed:", err);
            if (active) {
              setStatus("API response error");
              setError(
                err instanceof Error
                  ? err.message
                  : "An unknown error occurred while processing the frame."
              );
            }
          } finally {
            busyRef.current = false;
          }
        }, 500);

      } catch {
        if (active) {
          setStatus("Camera permission required");
          setError("Allow camera access in your browser, then reload this page.");
        }
      }
    };

    void startCamera();

    return () => {
      active = false;
      if (intervalId !== undefined) window.clearInterval(intervalId);
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    };
  }, [onMetrics]);

  const stopCamera = () => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraOn(false);
    setStatus("Camera stopped");
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.1 }}
      className="card-soft flex flex-col gap-4"
    >
      <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">
        Live View
      </h2>

      <div className="relative aspect-video rounded-3xl overflow-hidden bg-surface-cool inner-shadow ring-4 ring-pastel-blue/20">
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          className="absolute inset-0 h-full w-full object-cover -scale-x-100"
        />
        <canvas ref={canvasRef} className="hidden" />

        {!cameraOn && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-muted-foreground">
            <Camera size={32} className="opacity-50" />
            <span className="text-xs">Camera unavailable</span>
          </div>
        )}

        <div className="absolute top-4 right-4 flex items-center gap-2 rounded-full bg-background/80 px-3 py-1">
          {cameraOn ? <Circle size={9} className="fill-red-500 text-red-500" /> : <Square size={9} />}
          <span className="text-[10px] uppercase tracking-wider">
            {cameraOn ? "Live" : "Off"}
          </span>
        </div>

        <div className="absolute bottom-3 left-3 rounded-full bg-background/80 px-3 py-1 text-xs">
          {status}
        </div>
      </div>

      {error && <p role="status" className="text-xs text-destructive">{error}</p>}

      {cameraOn && (
        <button
          type="button"
          onClick={stopCamera}
          className="rounded-xl border px-3 py-2 text-sm hover:bg-muted"
        >
          Stop camera
        </button>
      )}

      <div className="flex flex-col gap-2 text-sm text-foreground/80">
        <div className="flex items-center gap-3">
          <span className={`h-2 w-2 rounded-full ${cameraOn ? "bg-pastel-sage" : "bg-muted-foreground"}`} />
          <Camera size={14} className="text-muted-foreground" />
          <span>Camera {cameraOn ? "on" : "off"}</span>
        </div>
        <div className="flex items-center gap-3">
          <span className={`h-2 w-2 rounded-full ${status === "Tracking" ? "bg-pastel-sage" : "bg-muted-foreground"}`} />
          <ScanFace size={14} className="text-muted-foreground" />
          <span>{status}</span>
        </div>
        <div className="flex items-center gap-3">
          <span className="h-2 w-2 rounded-full bg-muted-foreground" />
          <Eye size={14} className="text-muted-foreground" />
          <span>Eye measurements only; engagement model not connected</span>
        </div>
      </div>
    </motion.div>
  );
};

export default WebcamPanel;

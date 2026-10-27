import { useState } from "react";
import { motion } from "framer-motion";
import TopNav from "@/components/dashboard/TopNav";
import { Bell, Shield, Palette, Volume2, Globe } from "lucide-react";

const Toggle = ({ enabled, onToggle }: { enabled: boolean; onToggle: () => void }) => (
  <button
    onClick={onToggle}
    className={`relative w-11 h-6 rounded-full transition-colors duration-200 cursor-pointer ${
      enabled ? "bg-pastel-blue" : "bg-muted"
    }`}
  >
    <motion.div
      layout
      className="absolute top-0.5 w-5 h-5 rounded-full bg-card shadow-soft"
      style={{ left: enabled ? "calc(100% - 22px)" : "2px" }}
      transition={{ type: "spring", stiffness: 500, damping: 30 }}
    />
  </button>
);

const SettingsPage = () => {
  const [settings, setSettings] = useState({
    notifications: true,
    sounds: false,
    darkMode: false,
    autoSave: true,
    webcamPrivacy: true,
    language: "English",
  });

  const toggle = (key: keyof typeof settings) => {
    setSettings((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const settingSections = [
    {
      title: "Notifications",
      icon: Bell,
      items: [
        { label: "Push notifications", desc: "Get alerts about your learning progress", key: "notifications" as const },
        { label: "Sound effects", desc: "Play sounds for achievements and reminders", key: "sounds" as const },
      ],
    },
    {
      title: "Privacy",
      icon: Shield,
      items: [
        { label: "Webcam privacy mode", desc: "Blur your webcam feed when not in focus", key: "webcamPrivacy" as const },
        { label: "Auto-save sessions", desc: "Automatically save your learning sessions", key: "autoSave" as const },
      ],
    },
    {
      title: "Appearance",
      icon: Palette,
      items: [
        { label: "Dark mode", desc: "Switch to a darker color scheme", key: "darkMode" as const },
      ],
    },
  ];

  return (
    <div className="min-h-screen bg-background">
      <TopNav />
      <motion.main
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="max-w-3xl mx-auto px-4 sm:px-8 py-8"
      >
        <div className="mb-8">
          <h2 className="text-2xl font-semibold text-foreground">Settings ⚙️</h2>
          <p className="text-sm text-muted-foreground mt-1 leading-relaxed">
            Customize your learning experience.
          </p>
        </div>

        {/* Profile card */}
        <div className="card-soft flex items-center gap-4 mb-8">
          <div className="w-14 h-14 rounded-full bg-gradient-to-br from-pastel-pink to-pastel-blue flex items-center justify-center text-xl font-bold text-foreground">
            A
          </div>
          <div>
            <h3 className="text-base font-semibold text-foreground">Alex Johnson</h3>
            <p className="text-sm text-muted-foreground">alex.johnson@university.edu</p>
          </div>
          <button className="ml-auto text-sm font-medium text-foreground bg-muted hover:bg-pastel-blue/30 transition-colors rounded-2xl px-4 py-2 cursor-pointer">
            Edit Profile
          </button>
        </div>

        {/* Setting sections */}
        <div className="flex flex-col gap-6">
          {settingSections.map((section) => (
            <div key={section.title} className="card-soft">
              <div className="flex items-center gap-2 mb-5">
                <section.icon size={18} className="text-muted-foreground" />
                <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">
                  {section.title}
                </h3>
              </div>
              <div className="flex flex-col gap-4">
                {section.items.map((item) => (
                  <div key={item.key} className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-foreground">{item.label}</p>
                      <p className="text-xs text-muted-foreground">{item.desc}</p>
                    </div>
                    <Toggle
                      enabled={!!settings[item.key]}
                      onToggle={() => toggle(item.key)}
                    />
                  </div>
                ))}
              </div>
            </div>
          ))}

          {/* Language */}
          <div className="card-soft">
            <div className="flex items-center gap-2 mb-5">
              <Globe size={18} className="text-muted-foreground" />
              <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">
                Language
              </h3>
            </div>
            <select
              value={settings.language}
              onChange={(e) => setSettings((p) => ({ ...p, language: e.target.value }))}
              className="w-full bg-muted text-foreground text-sm rounded-2xl px-4 py-3 border-none outline-none cursor-pointer"
            >
              <option>English</option>
              <option>Spanish</option>
              <option>French</option>
              <option>German</option>
            </select>
          </div>
        </div>
      </motion.main>
    </div>
  );
};

export default SettingsPage;

import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useLocation, useNavigate } from "react-router-dom";
import { BookOpen, BarChart3, Settings, LayoutDashboard, User, LogOut } from "lucide-react";

const tabs = [
  { label: "Dashboard", icon: LayoutDashboard, path: "/" },
  { label: "Learning", icon: BookOpen, path: "/learning" },
  { label: "Analytics", icon: BarChart3, path: "/analytics" },
  { label: "Settings", icon: Settings, path: "/settings" },
];

const TopNav = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown on outside click
  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setDropdownOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  const getActiveTab = () => {
    const tab = tabs.find((t) => t.path === location.pathname);
    return tab?.label || "Dashboard";
  };

  return (
    <header className="sticky top-0 z-50 h-16 px-8 flex items-center justify-between bg-card/80 backdrop-blur-md">
      <h1
        className="text-lg font-bold tracking-tighter text-foreground cursor-pointer"
        onClick={() => navigate("/")}
      >
        <span className="text-pastel-blue">✦</span> Adaptive Learning
      </h1>

      <nav className="flex items-center gap-1">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = getActiveTab() === tab.label;
          return (
            <button
              key={tab.label}
              onClick={() => navigate(tab.path)}
              className={`relative text-sm font-medium px-4 py-2 rounded-full transition-colors flex items-center gap-2 cursor-pointer ${
                isActive
                  ? "text-foreground"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {isActive && (
                <motion.div
                  layoutId="activeTab"
                  className="absolute inset-0 bg-pastel-blue/30 rounded-full"
                  transition={{ type: "spring", stiffness: 400, damping: 30 }}
                />
              )}
              <Icon size={16} className="relative z-10" />
              <span className="relative z-10 hidden sm:inline">{tab.label}</span>
            </button>
          );
        })}
      </nav>

      {/* Profile */}
      <div className="relative" ref={dropdownRef}>
        <button
          onClick={() => setDropdownOpen((p) => !p)}
          className="w-9 h-9 rounded-full bg-gradient-to-br from-pastel-pink to-pastel-blue flex items-center justify-center text-sm font-semibold text-foreground cursor-pointer hover:scale-105 transition-transform"
        >
          A
        </button>

        <AnimatePresence>
          {dropdownOpen && (
            <motion.div
              initial={{ opacity: 0, y: 8, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 8, scale: 0.95 }}
              transition={{ duration: 0.15 }}
              className="absolute right-0 mt-2 w-48 bg-card rounded-2xl shadow-hover-effect overflow-hidden py-1"
            >
              <div className="px-4 py-3 border-b border-muted">
                <p className="text-sm font-semibold text-foreground">Alex Johnson</p>
                <p className="text-xs text-muted-foreground">Student</p>
              </div>
              <button
                onClick={() => {
                  setDropdownOpen(false);
                  navigate("/settings");
                }}
                className="w-full flex items-center gap-3 px-4 py-3 text-sm text-foreground hover:bg-muted transition-colors cursor-pointer"
              >
                <User size={14} className="text-muted-foreground" />
                View Profile
              </button>
              <button
                onClick={() => setDropdownOpen(false)}
                className="w-full flex items-center gap-3 px-4 py-3 text-sm text-foreground hover:bg-muted transition-colors cursor-pointer"
              >
                <LogOut size={14} className="text-muted-foreground" />
                Logout
              </button>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </header>
  );
};

export default TopNav;

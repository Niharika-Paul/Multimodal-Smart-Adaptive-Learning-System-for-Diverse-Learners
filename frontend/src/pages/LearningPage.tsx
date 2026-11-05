import { motion } from "framer-motion";
import TopNav from "@/components/dashboard/TopNav";
import { BookOpen, Play, FileText, Award, Clock } from "lucide-react";

const courses = [
  { title: "Introduction to Machine Learning", progress: 72, lessons: 12, icon: "🤖", color: "border-pastel-blue" },
  { title: "Data Visualization Fundamentals", progress: 45, lessons: 8, icon: "📊", color: "border-pastel-pink" },
  { title: "Statistics for Beginners", progress: 90, lessons: 15, icon: "📈", color: "border-pastel-sage" },
  { title: "Neural Networks Deep Dive", progress: 20, lessons: 10, icon: "🧠", color: "border-pastel-lavender" },
];

const LearningPage = () => (
  <div className="min-h-screen bg-background">
    <TopNav />
    <motion.main
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="max-w-7xl mx-auto px-4 sm:px-8 py-8"
    >
      <div className="mb-8">
        <h2 className="text-2xl font-semibold text-foreground">Your Learning Path 📚</h2>
        <p className="text-sm text-muted-foreground mt-1 leading-relaxed">
          Continue where you left off or explore new topics.
        </p>
      </div>

      {/* Active Courses */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        {courses.map((course) => (
          <motion.div
            key={course.title}
            whileHover={{ y: -4, transition: { duration: 0.2 } }}
            className={`card-soft border-t-4 ${course.color} cursor-pointer`}
          >
            <div className="flex items-start justify-between mb-4">
              <span className="text-2xl">{course.icon}</span>
              <span className="text-xs font-medium text-muted-foreground">
                {course.lessons} lessons
              </span>
            </div>
            <h3 className="text-base font-semibold text-foreground mb-3">{course.title}</h3>
            <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${course.progress}%` }}
                transition={{ duration: 0.8, delay: 0.2 }}
                className="h-full bg-pastel-blue rounded-full"
              />
            </div>
            <div className="flex items-center justify-between mt-2">
              <span className="text-xs text-muted-foreground tabular-nums">{course.progress}% complete</span>
              <button className="flex items-center gap-1 text-xs font-medium text-foreground hover:text-pastel-blue transition-colors">
                <Play size={12} /> Continue
              </button>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Recommended */}
      <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-4">Recommended For You</h3>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {[
          { title: "Adaptive Algorithms", icon: FileText, desc: "Learn how adaptive systems personalize content" },
          { title: "Study Techniques", icon: Award, desc: "Boost retention with proven methods" },
          { title: "Time Management", icon: Clock, desc: "Optimize your study schedule" },
        ].map((rec) => (
          <motion.div
            key={rec.title}
            whileHover={{ y: -3 }}
            className="card-soft cursor-pointer flex flex-col gap-3"
          >
            <rec.icon size={20} className="text-pastel-lavender" />
            <h4 className="text-sm font-semibold text-foreground">{rec.title}</h4>
            <p className="text-xs text-muted-foreground leading-relaxed">{rec.desc}</p>
          </motion.div>
        ))}
      </div>
    </motion.main>
  </div>
);

export default LearningPage;

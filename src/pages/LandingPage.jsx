import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { ScanEye, Ruler, Box, ArrowRight, CheckCircle, Zap, Lock } from 'lucide-react';
import VideoShowcase from '../components/VideoShowcase';
import { GeometricBackground } from '../components/ui/shape-landing-hero';

const LandingPage = () => {
    const navigate = useNavigate();
    const [activeTaskIndex, setActiveTaskIndex] = useState(0);

    // Auto-cycle effect
    useEffect(() => {
        const interval = setInterval(() => {
            setActiveTaskIndex((prev) => (prev + 1) % 3);
        }, 5000); // Cycle every 5 seconds
        return () => clearInterval(interval);
    }, []);

    const tasks = [
        {
            id: 'anomaly',
            title: 'High Precision',
            desc: 'Micron-level accuracy for critical components',
            icon: CheckCircle, // Changed icon to match "High Precision" vibe or keep original but styled
            path: '/anomaly',
            videoSrc: '/videos/landing-page-video.mp4'
        },
        {
            id: 'measurement',
            title: 'Real-Time Analysis',
            desc: 'Instant AI-powered defect detection',
            icon: Zap,
            path: '/measurement',
            videoSrc: '/videos/landing-page-video.mp4'
        },
        {
            id: 'assembly',
            title: 'Industry 4.0 Ready',
            desc: 'Seamless integration with modern workflows',
            icon: Lock,
            path: '/assembly',
            videoSrc: '/videos/landing-page-video.mp4'
        }
    ];

    return (
        <div className="min-h-screen bg-slate-950 flex flex-col items-center relative overflow-hidden font-sans selection:bg-neon selection:text-black">

            {/* Background Grid & Glows */}
            {/* Background Shapes */}
            <GeometricBackground />

            <div className="container mx-auto px-6 py-20 z-10 flex flex-col items-center relative">

                {/* Header Section */}
                <header className="text-center mb-20 max-w-4xl">
                    <motion.h1
                        initial={{ opacity: 0, y: 30 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="flex flex-col items-center justify-center mb-6"
                    >
                        <span className="text-5xl md:text-7xl font-bold text-white tracking-tight mb-2">
                            AI-Powered
                        </span>
                        <span className="text-5xl md:text-7xl font-extrabold text-neon tracking-tight drop-shadow-[0_0_15px_rgba(0,243,255,0.6)]">
                            Industrial Inspection
                        </span>
                    </motion.h1>

                    <motion.p
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.15 }}
                        className="text-xl md:text-2xl text-slate-400 font-medium max-w-2xl mx-auto"
                    >
                        Real-time defect detection and precision measurement for aerospace and quality assurance
                    </motion.p>
                </header>

                {/* Central Video Display */}
                <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.25 }}
                    className="w-full max-w-3xl aspect-video rounded-xl overflow-hidden border border-slate-800 shadow-[0_0_40px_rgba(0,243,255,0.15)] mb-20 relative bg-black group"
                >
                    <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent z-10 pointer-events-none" />
                    <video
                        src="/videos/landing-page-video.mp4"
                        autoPlay
                        loop
                        muted
                        playsInline
                        className="w-full h-full object-cover opacity-90 group-hover:opacity-100 transition-opacity duration-500"
                    />
                    {/* Decorative Corners */}
                    <div className="absolute top-4 left-4 w-8 h-8 border-l-2 border-t-2 border-neon/50 rounded-tl-sm z-20 opacity-60" />
                    <div className="absolute bottom-4 right-4 w-8 h-8 border-r-2 border-b-2 border-neon/50 rounded-br-sm z-20 opacity-60" />
                </motion.div>

                {/* Task/Feature Cards */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-8 w-full max-w-6xl mb-20">
                    {tasks.map((task, index) => (
                        <motion.div
                            key={task.id}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.3 + index * 0.1 }}
                            onMouseEnter={() => setActiveTaskIndex(index)}
                            onClick={() => navigate(task.path)}
                            className={`
                group relative p-8 rounded-2xl border transition-all duration-300 cursor-pointer overflow-hidden
                ${activeTaskIndex === index
                                    ? 'bg-slate-900/60 border-neon/30 shadow-[0_0_30px_rgba(0,243,255,0.1)]'
                                    : 'bg-slate-900/40 border-slate-800 hover:border-slate-700'}
                backdrop-blur-sm
              `}
                        >
                            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-neon to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />

                            <div className={`p-3 rounded-lg w-fit mb-6 transition-colors ${activeTaskIndex === index ? 'bg-neon/10 text-neon' : 'bg-slate-800 text-slate-400'}`}>
                                <task.icon className="w-8 h-8" strokeWidth={1.5} />
                            </div>

                            <h3 className={`text-2xl font-bold mb-3 transition-colors ${activeTaskIndex === index ? 'text-white' : 'text-slate-200'}`}>
                                {task.title}
                            </h3>
                            <p className="text-slate-400 leading-relaxed">
                                {task.desc}
                            </p>
                        </motion.div>
                    ))}
                </div>

                {/* CTA Button */}
                <motion.button
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.6 }}
                    onClick={() => navigate('/anomaly')}
                    className="group relative px-8 py-4 bg-neon text-slate-950 text-lg font-bold rounded-lg shadow-[0_0_20px_rgba(0,243,255,0.4)] hover:shadow-[0_0_40px_rgba(0,243,255,0.6)] hover:scale-105 transition-all duration-300 flex items-center gap-2"
                >
                    Start Inspection
                    <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                </motion.button>

                {/* Footer Status */}
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 1 }}
                    className="absolute bottom-8 flex items-center gap-2 text-slate-500 text-sm font-mono"
                >
                    <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                    Systems Ready
                </motion.div>

            </div>

            {/* Background Video Layer (Optional - kept subtle if needed, or removed for clean look) */}
            {/* We are focusing on cleaner text-based hero per the image, so I removed the large video showcase from the center and moved focus to the Cards + Header + CTA */}
        </div>
    );
};

export default LandingPage;

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Play } from 'lucide-react';

const VideoPlayer = ({ src, posterLabel }) => {
    const [hasError, setHasError] = useState(false);

    if (hasError || !src) {
        return (
            <div className="w-full h-full flex items-center justify-center bg-slate-800 relative overflow-hidden">
                <div className="absolute inset-0 bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 opacity-50" />
                <div className="z-10 text-center">
                    <div className="w-20 h-20 rounded-full bg-slate-700/50 flex items-center justify-center mx-auto mb-4 backdrop-blur-sm border border-slate-600">
                        <Play className="w-8 h-8 text-slate-400 ml-1" />
                    </div>
                    <h3 className="text-xl font-semibold text-slate-300">{posterLabel} Demo</h3>
                    <p className="text-slate-500 mt-2">Visual Simulation</p>
                </div>

                {/* Animated grid background for tech feel */}
                <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.03)_1px,transparent_1px)] bg-[size:40px_40px] opacity-20" />
            </div>
        );
    }

    return (
        <video
            className="w-full h-full object-cover"
            src={src}
            autoPlay
            loop
            muted
            playsInline
            onError={() => setHasError(true)}
        />
    );
};

const VideoShowcase = ({ activeIndex, tasks }) => {
    return (
        <div className="w-full h-full relative bg-black">
            <AnimatePresence mode="wait">
                <motion.div
                    key={activeIndex}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.5 }}
                    className="absolute inset-0 w-full h-full"
                >
                    <VideoPlayer
                        src={tasks[activeIndex].videoSrc}
                        posterLabel={tasks[activeIndex].title}
                    />

                    {/* Overlay Label */}
                    <div className="absolute top-6 left-6 z-20">
                        <div className="px-4 py-2 rounded bg-slate-950/50 backdrop-blur-md border border-slate-700/50 text-slate-200 text-sm font-medium flex items-center gap-2">
                            <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                            LIVE PREVIEW: {tasks[activeIndex].title.toUpperCase()}
                        </div>
                    </div>
                </motion.div>
            </AnimatePresence>

            {/* Decorative corners */}
            <div className="absolute top-4 left-4 w-4 h-4 border-l-2 border-t-2 border-cyan-500/50 rounded-tl-sm pointer-events-none" />
            <div className="absolute top-4 right-4 w-4 h-4 border-r-2 border-t-2 border-cyan-500/50 rounded-tr-sm pointer-events-none" />
            <div className="absolute bottom-4 left-4 w-4 h-4 border-l-2 border-b-2 border-cyan-500/50 rounded-bl-sm pointer-events-none" />
            <div className="absolute bottom-4 right-4 w-4 h-4 border-r-2 border-b-2 border-cyan-500/50 rounded-br-sm pointer-events-none" />
        </div>
    );
};

export default VideoShowcase;

import React from 'react';
import { motion } from 'framer-motion';
import { AlertTriangle, Activity, CheckCircle } from 'lucide-react';

const AnomalyDetection = () => {
    return (
        <div className="h-screen flex flex-col p-6 gap-6 w-full max-w-[1600px] mx-auto">
            {/* Header */}
            <div className="flex justify-between items-center">
                <div>
                    <h1 className="text-3xl font-bold text-slate-100">Anomaly Detection</h1>
                    <p className="text-slate-400">Real-time surface defect analysis</p>
                </div>
                <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2 px-4 py-2 bg-green-500/10 border border-green-500/20 rounded text-green-400 text-sm font-medium">
                        <Activity className="w-4 h-4" />
                        Model Running
                    </div>
                </div>
            </div>

            <div className="flex-1 flex gap-6 overflow-hidden">
                {/* Main Viewport */}
                <div className="flex-[3] bg-black rounded-lg border border-slate-800 relative overflow-hidden group">
                    {/* Simulated Camera Feed Background */}
                    <div className="absolute inset-0 bg-slate-900 flex items-center justify-center">
                        {/* Grid pattern */}
                        <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.05)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.05)_1px,transparent_1px)] bg-[size:50px_50px]" />
                        <p className="text-slate-600 font-mono">LIVE FEED SIGNAL</p>
                    </div>

                    {/* Simulated Defect Overlays */}
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: [0.4, 1, 0.4] }}
                        transition={{ duration: 2, repeat: Infinity }}
                        className="absolute top-[30%] left-[40%] w-32 h-32 border-2 border-red-500/80 rounded bg-red-500/10 flex items-start justify-start p-1"
                    >
                        <div className="text-[10px] bg-red-500 text-white px-1">SCRATCH 98%</div>
                    </motion.div>

                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: [0.2, 0.8, 0.2] }}
                        transition={{ duration: 3, repeat: Infinity, delay: 1 }}
                        className="absolute top-[60%] left-[20%] w-24 h-24 border-2 border-orange-500/80 rounded border-dashed bg-orange-500/10 flex items-start justify-start p-1"
                    >
                        <div className="text-[10px] bg-orange-500 text-white px-1">TEXTURE</div>
                    </motion.div>

                    {/* Scanning Line - Industrial Effect */}
                    <motion.div
                        animate={{ top: ['0%', '100%'] }}
                        transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
                        className="absolute left-0 w-full h-[2px] bg-cyan-500/50 shadow-[0_0_20px_rgba(34,211,238,0.5)] z-10"
                    />
                </div>

                {/* Right Panel - Inspector */}
                <div className="flex-1 bg-slate-900/50 border border-slate-800 rounded-lg p-6 flex flex-col">
                    <h3 className="text-lg font-semibold text-slate-200 mb-6 flex items-center gap-2">
                        <AlertTriangle className="w-5 h-5 text-yellow-500" /> Detected Anomalies
                    </h3>

                    <div className="space-y-4 flex-1">
                        <div className="p-4 bg-slate-800/50 rounded border-l-4 border-red-500">
                            <div className="flex justify-between items-start mb-2">
                                <span className="text-red-400 font-bold text-sm">CRITICAL</span>
                                <span className="text-slate-500 text-xs">00:01:23</span>
                            </div>
                            <h4 className="text-slate-200 font-medium">Surface Scratch</h4>
                            <p className="text-slate-400 text-sm mt-1">Found in Sector 4</p>
                            <div className="mt-3 flex items-center gap-2">
                                <div className="h-1 flex-1 bg-slate-700 rounded-full overflow-hidden">
                                    <div className="h-full w-[98%] bg-red-500" />
                                </div>
                                <span className="text-xs text-slate-300">98%</span>
                            </div>
                        </div>

                        <div className="p-4 bg-slate-800/50 rounded border-l-4 border-orange-500">
                            <div className="flex justify-between items-start mb-2">
                                <span className="text-orange-400 font-bold text-sm">WARNING</span>
                                <span className="text-slate-500 text-xs">00:01:21</span>
                            </div>
                            <h4 className="text-slate-200 font-medium">Texture Irregularity</h4>
                            <p className="text-slate-400 text-sm mt-1">Found in Sector 2</p>
                            <div className="mt-3 flex items-center gap-2">
                                <div className="h-1 flex-1 bg-slate-700 rounded-full overflow-hidden">
                                    <div className="h-full w-[78%] bg-orange-500" />
                                </div>
                                <span className="text-xs text-slate-300">78%</span>
                            </div>
                        </div>
                    </div>

                    <div className="mt-auto pt-6 border-t border-slate-800">
                        <div className="flex justify-between text-sm mb-2">
                            <span className="text-slate-400">Processed Items</span>
                            <span className="text-slate-200">1,248</span>
                        </div>
                        <div className="flex justify-between text-sm">
                            <span className="text-slate-400">Defect Rate</span>
                            <span className="text-red-400">1.2%</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default AnomalyDetection;

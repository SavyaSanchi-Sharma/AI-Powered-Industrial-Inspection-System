import React from 'react';
import { motion } from 'framer-motion';
import { CheckCircle, XCircle, RefreshCw, Box } from 'lucide-react';

const AssemblyVerification = () => {
    return (
        <div className="h-screen flex flex-col p-6 gap-6 w-full max-w-[1600px] mx-auto">
            {/* Header */}
            <div className="flex justify-between items-center">
                <div>
                    <h1 className="text-3xl font-bold text-slate-100">Assembly Verification</h1>
                    <p className="text-slate-400">Component integrity check</p>
                </div>
                <div className="flex items-center gap-4">
                    <div className="px-3 py-1 bg-slate-800 rounded border border-slate-700 text-xs text-slate-400">
                        SKU-9982-X
                    </div>
                </div>
            </div>

            <div className="flex-1 flex gap-6 overflow-hidden">

                {/* Left Panel: Reference Model & Live Feed & Controls */}
                <div className="flex-[3] flex flex-col gap-4">

                    <div className="flex-1 flex gap-4">
                        {/* Reference Image */}
                        <div className="flex-1 bg-slate-900 rounded-lg border border-slate-800 relative p-4 flex flex-col">
                            <span className="bg-slate-800 text-slate-400 text-xs px-2 py-1 rounded w-fit mb-4">REFERENCE MODEL</span>
                            <div className="flex-1 flex items-center justify-center opacity-60 grayscale">
                                {/* Placeholder Diagram */}
                                <div className="relative w-40 h-40 border-4 border-slate-600 rounded-full flex items-center justify-center">
                                    <div className="w-24 h-24 bg-slate-600 rounded"></div>
                                    <div className="absolute top-0 right-0 w-8 h-8 bg-slate-500 rounded-full border-2 border-slate-900"></div>
                                    <div className="absolute bottom-0 left-0 w-8 h-8 bg-slate-500 rounded-full border-2 border-slate-900"></div>
                                </div>
                            </div>
                        </div>

                        {/* Live Feed */}
                        <div className="flex-1 bg-black rounded-lg border border-slate-800 relative p-4 flex flex-col">
                            <span className="bg-red-500/20 text-red-400 text-xs px-2 py-1 rounded w-fit mb-4 border border-red-500/30 flex items-center gap-2">
                                <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" /> LIVE FEED
                            </span>
                            <div className="flex-1 flex items-center justify-center relative">
                                {/* Simulated Live Object - Modified to show missing part */}
                                <div className="relative w-40 h-40 border-4 border-slate-400 rounded-full flex items-center justify-center">
                                    <div className="w-24 h-24 bg-slate-400 rounded"></div>
                                    <div className="absolute top-0 right-0 w-8 h-8 bg-slate-500 rounded-full border-2 border-slate-900"></div>
                                    {/* Missing Part Indicator */}
                                    <motion.div
                                        animate={{ opacity: [0.5, 1, 0.5] }}
                                        transition={{ duration: 1, repeat: Infinity }}
                                        className="absolute bottom-0 left-0 w-8 h-8 border-2 border-red-500 border-dashed rounded-full bg-red-500/20 flex items-center justify-center"
                                    >
                                        <XCircle className="w-4 h-4 text-red-500" />
                                    </motion.div>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Control Buttons */}
                    <div className="h-20 bg-slate-900/50 border border-slate-800 rounded-lg flex items-center justify-center gap-6">
                        <button className="px-8 py-3 bg-cyan-500 hover:bg-cyan-400 text-black font-semibold rounded shadow shadow-cyan-500/20 transition-all flex items-center gap-2">
                            <RefreshCw className="w-5 h-5" /> Scan
                        </button>
                        <button className="px-8 py-3 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-600 rounded transition-all flex items-center gap-2">
                            <Box className="w-5 h-5 text-slate-400" /> Calibrate
                        </button>
                    </div>

                </div>

                {/* Right Panel - Missing Parts Log */}
                <div className="flex-1 bg-slate-900/50 border border-slate-800 rounded-lg p-6 flex flex-col">
                    <h3 className="text-lg font-semibold text-slate-200 mb-6 flex items-center gap-2">
                        <Box className="w-5 h-5 text-red-500" /> Missing Parts
                    </h3>

                    <div className="space-y-4 flex-1 overflow-y-auto">
                        {/* Log Item Card */}
                        <div className="p-4 bg-slate-800/50 rounded border-l-4 border-red-500 animate-in fade-in slide-in-from-right-4 duration-500">
                            <div className="flex justify-between items-start mb-2">
                                <span className="text-red-400 font-bold text-sm">MISSING</span>
                                <span className="text-slate-500 text-xs">ID: P-04</span>
                            </div>
                            <h4 className="text-slate-200 font-medium">Locking Bolt</h4>
                            <div className="mt-2 flex items-baseline justify-between">
                                <span className="text-xs text-slate-400">Position: Lower Left</span>
                                <div className="flex items-center gap-1 text-red-400 text-xs">
                                    <XCircle className="w-3 h-3" /> Critical
                                </div>
                            </div>
                            <div className="mt-3 h-1 w-full bg-slate-700 rounded-full overflow-hidden">
                                <div className="h-full bg-red-500 w-full" style={{ width: '100%' }} />
                            </div>
                        </div>
                    </div>

                    <div className="mt-auto pt-6 border-t border-slate-800">
                        <div className="flex justify-between text-sm mb-4">
                            <span className="text-slate-400">Items Scanned</span>
                            <span className="text-slate-200">1</span>
                        </div>

                        <div className="grid grid-cols-2 gap-3">
                            <button className="py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 rounded text-xs transition-colors flex items-center justify-center gap-2">
                                Export CSV
                            </button>
                            <button className="py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 rounded text-xs transition-colors flex items-center justify-center gap-2">
                                Export JSON
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default AssemblyVerification;

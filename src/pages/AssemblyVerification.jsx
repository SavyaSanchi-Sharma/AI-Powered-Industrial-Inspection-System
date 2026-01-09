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

                {/* Split View */}
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

                    {/* Status Banner */}
                    <div className="h-24 bg-red-500/10 border border-red-500/50 rounded-lg flex items-center justify-between px-8">
                        <div className="flex items-center gap-4">
                            <div className="w-12 h-12 rounded-full bg-red-500 flex items-center justify-center">
                                <XCircle className="w-8 h-8 text-white" />
                            </div>
                            <div>
                                <h2 className="text-2xl font-bold text-red-500">ASSEMBLY FAILED</h2>
                                <p className="text-red-400/70">Critical component missing</p>
                            </div>
                        </div>
                        <button className="px-6 py-2 bg-slate-800 hover:bg-slate-700 border border-slate-600 rounded text-slate-200 flex items-center gap-2 transition-colors">
                            <RefreshCw className="w-4 h-4" /> Re-Scan
                        </button>
                    </div>

                </div>

                {/* Right Panel - Checklist */}
                <div className="flex-1 bg-slate-900/50 border border-slate-800 rounded-lg p-6 flex flex-col">
                    <h3 className="text-lg font-semibold text-slate-200 mb-6 flex items-center gap-2">
                        <Box className="w-5 h-5 text-cyan-500" /> Parts Checklist
                    </h3>

                    <div className="space-y-3">
                        <div className="flex items-center gap-3 p-3 bg-slate-800/30 rounded border border-green-500/20">
                            <CheckCircle className="w-5 h-5 text-green-500" />
                            <span className="text-slate-300">Base Housing</span>
                        </div>
                        <div className="flex items-center gap-3 p-3 bg-slate-800/30 rounded border border-green-500/20">
                            <CheckCircle className="w-5 h-5 text-green-500" />
                            <span className="text-slate-300">Main Rotor</span>
                        </div>
                        <div className="flex items-center gap-3 p-3 bg-red-500/10 rounded border border-red-500/30">
                            <XCircle className="w-5 h-5 text-red-500" />
                            <span className="text-red-400 font-medium">Locking Bolt</span>
                            <span className="ml-auto text-xs text-red-400 bg-red-500/10 px-2 py-0.5 rounded">MISSING</span>
                        </div>
                        <div className="flex items-center gap-3 p-3 bg-slate-800/30 rounded border border-green-500/20">
                            <CheckCircle className="w-5 h-5 text-green-500" />
                            <span className="text-slate-300">Upper Cover</span>
                        </div>
                    </div>

                    <div className="mt-auto p-4 bg-slate-800 rounded-lg">
                        <h4 className="text-xs text-slate-500 uppercase tracking-wider mb-2">Confidence Score</h4>
                        <div className="flex items-end gap-1">
                            <span className="text-3xl font-bold text-slate-200">99.9</span>
                            <span className="text-lg text-slate-500 mb-1">%</span>
                        </div>
                        <div className="w-full bg-slate-700 h-1 mt-2 rounded-full overflow-hidden">
                            <div className="w-[99.9%] h-full bg-cyan-500" />
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default AssemblyVerification;

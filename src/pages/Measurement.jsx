import React from 'react';
import { motion } from 'framer-motion';
import { Ruler, Check, Settings } from 'lucide-react';

const Measurement = () => {
    const [measurementCount, setMeasurementCount] = React.useState(2);
    const [measurements, setMeasurements] = React.useState([
        { name: 'Width', value: '256.0', unit: 'mm' },
        { name: 'Height', value: '160.0', unit: 'mm' }
    ]);

    const handleCountChange = (e) => {
        const count = parseInt(e.target.value) || 0;
        setMeasurementCount(count);

        // Adjust array size
        if (count > measurements.length) {
            const newItems = Array(count - measurements.length).fill({ name: '', value: '', unit: 'mm' });
            setMeasurements([...measurements, ...newItems]);
        } else {
            setMeasurements(measurements.slice(0, count));
        }
    };

    const handleMeasurementChange = (index, field, value) => {
        const newMeasurements = [...measurements];
        newMeasurements[index] = { ...newMeasurements[index], [field]: value };
        setMeasurements(newMeasurements);
    };

    return (
        <div className="h-screen flex flex-col p-6 gap-6 w-full max-w-[1600px] mx-auto">
            {/* Header */}
            <div className="flex justify-between items-center">
                <div>
                    <h1 className="text-3xl font-bold text-slate-100">Precision Measurement</h1>
                    <p className="text-slate-400">Automated dimensional analysis</p>
                </div>
                <div className="flex items-center gap-4">
                    <div className="text-xs text-slate-500 font-mono">CALIBRATED: ±1mm</div>
                </div>
            </div>

            <div className="flex-1 flex gap-6 overflow-hidden">
                {/* Main Viewport */}
                <div className="flex-[3] bg-black rounded-lg border border-slate-800 relative overflow-hidden flex items-center justify-center">

                    {/* Simulated Object (Placeholder Box) */}
                    <div className="w-64 h-40 bg-slate-800 rounded border border-slate-700 relative shadow-2xl">
                        {/* Texture/Detail */}
                        <div className="absolute top-4 right-4 w-2 h-2 rounded-full bg-slate-600" />
                        <div className="absolute bottom-4 left-4 w-full h-1 bg-slate-700/50" />
                    </div>

                    {/* SVG Overlay layer */}
                    <svg className="absolute inset-0 w-full h-full pointer-events-none">
                        <defs>
                            <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
                                <path d="M0,0 L0,6 L9,3 z" fill="#22d3ee" />
                            </marker>
                        </defs>

                        {/* Horizontal Measure */}
                        <motion.path
                            d="M 500 480 L 750 480" // Adjust coords based on container approximation or use percentage in CSS
                            // Since I can't know exact pixels, I'll use percentages via CSS/Inline styles usually, but SVG path needs coords.
                            // I'll cheat and put the labels/lines in absolute divs relative to the center object.
                            className="hidden"
                        />
                    </svg>

                    {/* HTML Overlay wrappers for responsiveness simplified */}

                    {/* Width Measurement */}
                    <motion.div
                        initial={{ width: 0, opacity: 0 }}
                        animate={{ width: 256, opacity: 1 }} // 64 * 4 (w-64 is 16rem = 256px) - wait, w-64 is 16rem = 256px
                        transition={{ duration: 1, delay: 0.5 }}
                        className="absolute mt-52 w-64 h-px bg-cyan-400 flex flex-col items-center justify-center"
                    >
                        <div className="w-full h-2 border-l border-r border-cyan-400 absolute -top-1" />
                        <span className="bg-black text-cyan-400 text-xs px-1 font-mono mt-4">256.0 mm</span>
                    </motion.div>

                    {/* Height Measurement */}
                    <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 160, opacity: 1 }} // h-40 is 10rem = 160px
                        transition={{ duration: 1, delay: 0.8 }}
                        className="absolute ml-[280px] h-40 w-px bg-cyan-400 flex flex-row items-center justify-center"
                    >
                        <div className="h-full w-2 border-t border-b border-cyan-400 absolute -left-1" />
                        <span className="bg-black text-cyan-400 text-xs px-1 font-mono ml-8 whitespace-nowrap">160.0 mm</span>
                    </motion.div>

                </div>

                <div className="flex-1 bg-slate-900/50 border border-slate-800 rounded-lg p-6 flex flex-col overflow-y-auto">
                    <h3 className="text-lg font-semibold text-slate-200 mb-6 flex items-center gap-2">
                        <Ruler className="w-5 h-5 text-cyan-500" /> Manual Input
                    </h3>

                    {/* Input Form */}
                    <div className="space-y-4 mb-8">
                        <div>
                            <label className="block text-xs text-slate-500 mb-1 uppercase tracking-wider">Number of Measurements</label>
                            <input
                                type="number"
                                min="1"
                                max="10"
                                value={measurementCount}
                                onChange={handleCountChange}
                                className="w-full bg-slate-800 border border-slate-700 rounded p-2 text-slate-200 focus:border-cyan-500 focus:outline-none transition-colors"
                            />
                        </div>

                        <div className="pt-4 border-t border-slate-800/50">
                            <div className="grid grid-cols-3 gap-2 mb-2">
                                <label className="text-[10px] text-slate-500 uppercase">Name</label>
                                <label className="text-[10px] text-slate-500 uppercase">Value</label>
                                <label className="text-[10px] text-slate-500 uppercase">Unit</label>
                            </div>

                            {measurements.map((measure, index) => (
                                <div key={index} className="grid grid-cols-3 gap-2 mb-2">
                                    <input
                                        value={measure.name}
                                        onChange={(e) => handleMeasurementChange(index, 'name', e.target.value)}
                                        className="bg-slate-800 border border-slate-700 rounded p-2 text-sm text-slate-200 focus:border-cyan-500/50 outline-none"
                                        placeholder="Name"
                                    />
                                    <input
                                        value={measure.value}
                                        onChange={(e) => handleMeasurementChange(index, 'value', e.target.value)}
                                        className="bg-slate-800 border border-slate-700 rounded p-2 text-sm text-slate-200 focus:border-cyan-500/50 outline-none"
                                        placeholder="0.00"
                                    />
                                    <input
                                        value={measure.unit}
                                        onChange={(e) => handleMeasurementChange(index, 'unit', e.target.value)}
                                        className="bg-slate-800 border border-slate-700 rounded p-2 text-sm text-slate-200 focus:border-cyan-500/50 outline-none"
                                        placeholder="mm"
                                    />
                                </div>
                            ))}
                        </div>

                        <button className="w-full py-2 bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-400 border border-cyan-500/50 rounded text-sm transition-colors flex items-center justify-center gap-2">
                            <Settings className="w-4 h-4" /> Update Configuration
                        </button>
                    </div>

                    <h3 className="text-lg font-semibold text-slate-200 mb-4 flex items-center gap-2">
                        <Settings className="w-5 h-5 text-slate-500" /> System Data
                    </h3>

                    <div className="space-y-1">
                        <div className="flex justify-between p-3 border-b border-slate-800">
                            <span className="text-slate-400">Object ID</span>
                            <span className="text-slate-200 font-mono">#A7-22</span>
                        </div>
                        {measurements.map((m, i) => (
                            <div key={i} className="flex justify-between p-3 border-b border-slate-800 even:bg-slate-800/20">
                                <span className="text-slate-400">{m.name || '-'}</span>
                                <span className="text-cyan-400 font-mono font-bold">
                                    {m.value || '0.00'} {m.unit}
                                </span>
                            </div>
                        ))}
                    </div>

                    <div className="mt-8 p-4 bg-green-500/5 border border-green-500/20 rounded-lg text-center">
                        <Check className="w-8 h-8 text-green-500 mx-auto mb-2" />
                        <div className="text-green-400 font-medium">WITHIN TOLERANCE</div>
                        <div className="text-green-600/70 text-xs mt-1">QC PASSED</div>
                    </div>

                    <div className="mt-auto pt-4">
                        <button className="w-full py-3 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded text-sm transition-colors border border-slate-700">
                            Export Report
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Measurement;

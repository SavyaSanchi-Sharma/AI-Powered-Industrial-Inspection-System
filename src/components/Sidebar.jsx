import React from 'react';
import { NavLink } from 'react-router-dom';
import { Home, ScanEye, Ruler, Box } from 'lucide-react';
import clsx from 'clsx';

const NavItem = ({ to, icon: Icon, label }) => (
    <NavLink
        to={to}
        className={({ isActive }) =>
            clsx(
                "flex flex-col items-center justify-center p-4 py-6 transition-all duration-300 border-l-2",
                isActive
                    ? "bg-slate-900 border-neon text-neon"
                    : "border-transparent text-slate-400 hover:text-slate-200 hover:bg-slate-900/50"
            )
        }
    >
        <Icon size={24} className="mb-1" />
        <span className="text-[10px] uppercase tracking-wider font-semibold">{label}</span>
    </NavLink>
);

const Sidebar = () => {
    return (
        <aside className="w-24 bg-slate-950 border-r border-slate-800 flex flex-col z-50">
            <div className="flex-1 flex flex-col pt-4">
                <NavItem to="/" icon={Home} label="Home" />
                <div className="h-px bg-slate-800 my-2 mx-4" />
                <NavItem to="/anomaly" icon={ScanEye} label="Anomaly" />
                <NavItem to="/measurement" icon={Ruler} label="Measure" />
                <NavItem to="/assembly" icon={Box} label="Assembly" />
            </div>
            <div className="p-4 text-center">
                <div className="w-2 h-2 rounded-full bg-green-500 mx-auto animate-pulseShadow" />
                <span className="text-[10px] text-slate-500 mt-2 block">ONLINE</span>
            </div>
        </aside>
    );
};

export default Sidebar;

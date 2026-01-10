import React from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import LandingPage from './pages/LandingPage';
import AnomalyDetection from './pages/AnomalyDetection';
import Measurement from './pages/Measurement';
import AssemblyVerification from './pages/AssemblyVerification';
import { DemoHeroGeometric } from './components/DemoHeroGeometric';
import Sidebar from './components/Sidebar';

/* Layout wrapper to handle sidebar visibility */
const AppLayout = ({ children }) => {
  const location = useLocation();
  const isLanding = location.pathname === '/' || location.pathname === '/style-demo';

  return (
    <div className="flex min-h-screen bg-slate-950 text-slate-50 font-sans">
      {!isLanding && <Sidebar />}
      <main className="flex-1 relative overflow-hidden">
        {children}
      </main>
    </div>
  );
};

function App() {
  return (
    <Router>
      <AppLayout>
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/style-demo" element={<DemoHeroGeometric />} />
          <Route path="/anomaly" element={<AnomalyDetection />} />
          <Route path="/measurement" element={<Measurement />} />
          <Route path="/assembly" element={<AssemblyVerification />} />
        </Routes>
      </AppLayout>
    </Router>
  );
}

export default App;

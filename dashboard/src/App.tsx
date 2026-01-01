/**
 * Main Application Component
 * ==========================
 */

import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Dashboard from './components/layout/Dashboard';
import { TruthEngine } from './pages/TruthEngine';

const App: React.FC = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/truth-engine" element={<TruthEngine />} />
      </Routes>
    </Router>
  );
};

export default App;

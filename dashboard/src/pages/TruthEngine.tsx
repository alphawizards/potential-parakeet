/**
 * Truth Engine Dashboard Page
 * ===========================
 * Institutional-grade strategy validation dashboard.
 * Displays DSR/PSR metrics, graveyard statistics, and forensic charts.
 */

import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import {
    MOCK_STRATEGIES
} from '../components/truth-engine/mockData';
import { ValidityCard } from '../components/truth-engine/ValidityCard';
import { ForensicCharts } from '../components/truth-engine/ForensicCharts';
import {
    Shield,
    ArrowLeft
} from 'lucide-react';

export const TruthEngine: React.FC = () => {
    // Default to the first strategy (Valid)
    const [selectedStrategyId, setSelectedStrategyId] = useState<string>(MOCK_STRATEGIES[0].id);

    const selectedStrategy = MOCK_STRATEGIES.find(s => s.id === selectedStrategyId) || MOCK_STRATEGIES[0];

    return (
        <div className="min-h-screen bg-gray-50 text-gray-900">
            {/* Header */}
            <header className="bg-white border-b border-gray-200 px-6 py-4 sticky top-0 z-50 shadow-sm">
                <div className="max-w-7xl mx-auto flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <Link to="/" className="text-gray-500 hover:text-gray-700 transition-colors">
                            <ArrowLeft className="w-6 h-6" />
                        </Link>
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center">
                                <Shield className="w-6 h-6 text-white" />
                            </div>
                            <div>
                                <h1 className="text-xl font-bold text-gray-900">Truth Engine</h1>
                                <p className="text-xs text-gray-500">Statistical Validity & Forensics</p>
                            </div>
                        </div>
                    </div>

                    {/* Strategy Selector */}
                    <div>
                        <select
                            value={selectedStrategyId}
                            onChange={(e) => setSelectedStrategyId(e.target.value)}
                            className="block w-64 pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                        >
                            {MOCK_STRATEGIES.map(s => (
                                <option key={s.id} value={s.id}>{s.name}</option>
                            ))}
                        </select>
                    </div>
                </div>
            </header>

            {/* Main Content */}
            <main className="max-w-7xl mx-auto px-6 py-8 space-y-8">
                {/* Validity Card */}
                <section>
                    <ValidityCard
                        metrics={selectedStrategy.validity}
                        sharpe={selectedStrategy.efficiency.sharpe}
                    />
                </section>

                {/* Forensic Charts */}
                <section>
                    <h2 className="text-lg font-bold text-gray-900 mb-4">Forensic Analysis</h2>
                    <ForensicCharts
                        icDecay={selectedStrategy.ic_decay}
                        attribution={selectedStrategy.attribution}
                        executionSurface={selectedStrategy.execution_surface}
                    />
                </section>
            </main>
        </div>
    );
};

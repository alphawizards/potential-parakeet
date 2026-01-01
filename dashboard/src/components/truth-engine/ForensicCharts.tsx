import React from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, Legend, ScatterChart, Scatter
} from 'recharts';
import { ICDecay, Attribution, ExecutionData } from '../../types/strategy';

interface ForensicChartsProps {
  icDecay?: ICDecay[];
  attribution?: Attribution;
  executionSurface?: ExecutionData[];
}

export const ForensicCharts: React.FC<ForensicChartsProps> = ({
  icDecay = [],
  attribution,
  executionSurface = []
}) => {

  // Format attribution for Bar Chart
  const attributionData = attribution ? [
    {
      name: 'Return Components',
      'Market Beta': attribution.market_beta,
      'Style Factors': attribution.style_factors,
      'Pure Alpha': attribution.idiosyncratic_alpha,
    }
  ] : [];

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* IC Decay Chart */}
      <div className="bg-white p-4 rounded-xl border border-gray-200 shadow-sm">
        <h3 className="text-sm font-semibold text-gray-700 mb-4">Information Coefficient (IC) Decay</h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={icDecay}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="horizon" label={{ value: 'Horizon (Days)', position: 'insideBottom', offset: -5 }} />
              <YAxis label={{ value: 'IC', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Line type="monotone" dataKey="ic" stroke="#2563eb" strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Factor Attribution Chart */}
      <div className="bg-white p-4 rounded-xl border border-gray-200 shadow-sm">
        <h3 className="text-sm font-semibold text-gray-700 mb-4">Factor Attribution</h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={attributionData}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="Market Beta" stackId="a" fill="#94a3b8" />
              <Bar dataKey="Style Factors" stackId="a" fill="#60a5fa" />
              <Bar dataKey="Pure Alpha" stackId="a" fill="#10b981" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Execution Surface Chart */}
      <div className="bg-white p-4 rounded-xl border border-gray-200 shadow-sm">
        <h3 className="text-sm font-semibold text-gray-700 mb-4">Execution Surface (Slippage vs VIX)</h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" dataKey="vix" name="VIX" label={{ value: 'VIX', position: 'insideBottom', offset: -5 }} />
              <YAxis type="number" dataKey="slippage_bps" name="Slippage" unit="bps" label={{ value: 'Slippage (bps)', angle: -90, position: 'insideLeft' }} />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} />
              <Scatter name="Executions" data={executionSurface} fill="#ef4444" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

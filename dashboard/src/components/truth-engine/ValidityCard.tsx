import React from 'react';
import { ShieldCheck, AlertTriangle } from 'lucide-react';
import { StrategyMetrics } from '../../types/strategy';

interface ValidityCardProps {
  metrics: StrategyMetrics['validity'];
  sharpe: number;
}

export const ValidityCard: React.FC<ValidityCardProps> = ({ metrics, sharpe }) => {
  const { dsr, num_trials } = metrics;

  // Logic: Overfit if DSR < 0.5 OR Num Trials > 100 (while sharpe < 1.5)
  // Note: The prompt says "while sharpe < 1.5", implying high sharpe might excuse high trials?
  // Let's implement strictly: (dsr < 0.5) OR (num_trials > 100) -> Overfit Warning.
  // Wait, prompt says: "If dsr < 0.5 OR num_trials > 100 (while sharpe < 1.5)"
  // This likely means: if dsr < 0.5 OR (num_trials > 100 AND sharpe < 1.5)

  const isOverfit = dsr < 0.5 || (num_trials > 100 && sharpe < 1.5);

  return (
    <div className={`p-6 rounded-xl border ${isOverfit ? 'bg-red-50 border-red-200' : 'bg-emerald-50 border-emerald-200'}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className={`text-lg font-bold ${isOverfit ? 'text-red-800' : 'text-emerald-800'}`}>
          Strategy Validity
        </h3>
        {isOverfit ? (
          <div className="flex items-center gap-2 px-3 py-1 bg-red-100 text-red-700 rounded-full border border-red-200">
            <AlertTriangle className="w-4 h-4" />
            <span className="text-sm font-bold">OVERFIT WARNING</span>
          </div>
        ) : (
          <div className="flex items-center gap-2 px-3 py-1 bg-emerald-100 text-emerald-700 rounded-full border border-emerald-200">
            <ShieldCheck className="w-4 h-4" />
            <span className="text-sm font-bold">VALID</span>
          </div>
        )}
      </div>

      <div className="grid grid-cols-2 gap-6">
        <div>
          <p className="text-sm text-gray-500 mb-1">Deflated Sharpe (DSR)</p>
          <div className="flex items-baseline gap-2">
            <span className={`text-2xl font-bold ${dsr < 0.5 ? 'text-red-600' : 'text-gray-900'}`}>
              {dsr.toFixed(2)}
            </span>
            <span className="text-xs text-gray-400">Target: &gt; 0.5</span>
          </div>
        </div>
        <div>
          <p className="text-sm text-gray-500 mb-1">The Graveyard (Trials)</p>
          <div className="flex items-baseline gap-2">
            <span className={`text-2xl font-bold ${num_trials > 100 ? 'text-orange-600' : 'text-gray-900'}`}>
              {num_trials}
            </span>
            <span className="text-xs text-gray-400">Target: &lt; 100</span>
          </div>
        </div>
      </div>
    </div>
  );
};

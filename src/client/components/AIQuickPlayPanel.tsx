import React, { useState } from 'react';
import { BoardType } from '../../shared/types/game';
import { Select } from './ui/Select';
import {
  AIQuickPlayOption,
  BOARD_DISPLAY_NAMES,
  TIER_COLORS,
  getOptionsForConfig,
  DifficultyTier,
} from '../config/aiQuickPlay';

interface AIQuickPlayPanelProps {
  onStartGame: (option: AIQuickPlayOption) => void;
  isLoading: boolean;
}

function DifficultyCard({
  option,
  onClick,
  disabled,
}: {
  option: AIQuickPlayOption;
  onClick: () => void;
  disabled: boolean;
}) {
  const colors = TIER_COLORS[option.difficultyTier];

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`
        p-4 rounded-xl border-2 ${colors.border} ${colors.bg}
        transition-all duration-200 text-left
        hover:scale-[1.02] hover:shadow-lg
        focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:ring-offset-2 focus:ring-offset-slate-900
        disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100
      `}
    >
      <div className={`font-semibold ${colors.text}`}>{option.displayName}</div>
      <div className="text-xs text-slate-400 mt-1">D{option.difficulty}</div>
      <div className="text-xs text-slate-300 mt-2">{option.description}</div>
      <div className="text-[10px] text-slate-500 mt-1">~{option.estimatedElo} Elo</div>
    </button>
  );
}

export function AIQuickPlayPanel({ onStartGame, isLoading }: AIQuickPlayPanelProps) {
  const [selectedBoard, setSelectedBoard] = useState<BoardType>('square8');
  const [selectedPlayers, setSelectedPlayers] = useState<number>(2);

  const filteredOptions = getOptionsForConfig(selectedBoard, selectedPlayers);

  // Sort options by difficulty tier order
  const tierOrder: DifficultyTier[] = ['easy', 'medium', 'hard', 'expert'];
  const sortedOptions = [...filteredOptions].sort(
    (a, b) => tierOrder.indexOf(a.difficultyTier) - tierOrder.indexOf(b.difficultyTier)
  );

  return (
    <div className="bg-slate-800/70 rounded-xl p-6 border border-emerald-500/30 shadow-lg">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-white flex items-center gap-2">
          Play vs AI
          <span className="text-xs px-2 py-0.5 bg-emerald-500/20 text-emerald-300 rounded-full border border-emerald-500/30">
            Instant
          </span>
        </h2>
        <p className="text-xs text-slate-400">Choose difficulty and start immediately</p>
      </div>

      {/* Board/Player selector */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div>
          <label htmlFor="ai-quick-board" className="block text-sm font-medium text-slate-300 mb-1">
            Board
          </label>
          <Select
            id="ai-quick-board"
            value={selectedBoard}
            onChange={(e) => setSelectedBoard(e.target.value as BoardType)}
          >
            {Object.entries(BOARD_DISPLAY_NAMES).map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </Select>
        </div>
        <div>
          <label
            htmlFor="ai-quick-players"
            className="block text-sm font-medium text-slate-300 mb-1"
          >
            Players
          </label>
          <Select
            id="ai-quick-players"
            value={selectedPlayers}
            onChange={(e) => setSelectedPlayers(Number(e.target.value))}
          >
            <option value={2}>1 vs 1 AI</option>
            <option value={3}>1 vs 2 AI</option>
            <option value={4}>1 vs 3 AI</option>
          </Select>
        </div>
      </div>

      {/* Difficulty cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {sortedOptions.map((option) => (
          <DifficultyCard
            key={option.id}
            option={option}
            onClick={() => onStartGame(option)}
            disabled={isLoading}
          />
        ))}
      </div>

      {/* Helper text */}
      <p className="mt-4 text-xs text-slate-500 text-center">
        AI games are unrated. Elo estimates are approximate based on AI training data.
      </p>
    </div>
  );
}

export type { AIQuickPlayOption };

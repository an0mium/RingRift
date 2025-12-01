/**
 * State persistence utilities for sandbox mode.
 *
 * Provides functions to:
 * - Save current game state as a custom scenario
 * - Export scenarios to downloadable JSON files
 * - Import scenarios from uploaded JSON files
 */

import type { GameState, BoardType } from '../../shared/types/game';
import type { LoadableScenario, ScenarioCategory } from './scenarioTypes';
import { serializeGameState } from '../../shared/engine/contracts/serialization';
import { saveCustomScenario } from './scenarioLoader';

/**
 * Metadata to include when saving a game state.
 */
export interface SavedGameMetadata {
  /** User-provided name for the saved state */
  name: string;
  /** User-provided description */
  description?: string | undefined;
  /** Optional category override */
  category?: ScenarioCategory | undefined;
  /** Optional tags */
  tags?: string[] | undefined;
}

/**
 * Generate a unique ID for a custom scenario.
 */
function generateScenarioId(): string {
  const timestamp = Date.now();
  const random = Math.random().toString(36).substring(2, 9);
  return `custom_${timestamp}_${random}`;
}

/**
 * Infer the category based on game phase.
 */
function inferCategoryFromPhase(phase: string): ScenarioCategory {
  switch (phase) {
    case 'ring_placement':
      return 'placement';
    case 'movement':
      return 'movement';
    case 'chain_capture':
      return 'chain_capture';
    case 'process_line':
    case 'choose_line_reward':
      return 'line_processing';
    case 'process_territory_region':
      return 'territory_processing';
    default:
      return 'custom';
  }
}

/**
 * Save the current game state as a custom scenario.
 *
 * @param gameState - The current game state to save
 * @param metadata - User-provided metadata for the saved state
 * @returns The created LoadableScenario
 */
export function saveCurrentGameState(
  gameState: GameState,
  metadata: SavedGameMetadata
): LoadableScenario {
  const serializedState = serializeGameState(gameState);
  const id = generateScenarioId();

  const turnNumber = (gameState.moveHistory?.length ?? 0) + 1;
  const defaultDescription = `Saved at turn ${turnNumber}, ${gameState.currentPhase} phase`;

  const scenario: LoadableScenario = {
    id,
    name: metadata.name || `Saved Game - ${new Date().toLocaleDateString()}`,
    description: metadata.description || defaultDescription,
    category: metadata.category || inferCategoryFromPhase(gameState.currentPhase),
    tags: metadata.tags || ['saved', gameState.currentPhase],
    boardType: gameState.boardType,
    playerCount: gameState.players.length,
    createdAt: new Date().toISOString(),
    source: 'custom',
    state: serializedState,
  };

  saveCustomScenario(scenario);
  return scenario;
}

/**
 * Export a scenario to a downloadable JSON file.
 *
 * @param scenario - The scenario to export
 */
export function exportScenarioToFile(scenario: LoadableScenario): void {
  const json = JSON.stringify(scenario, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);

  // Create safe filename
  const safeName = scenario.name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_|_$/g, '');
  const filename = `ringrift_scenario_${safeName}.json`;

  // Create download link and trigger click
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();

  // Cleanup
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/**
 * Export the current game state directly to a file.
 *
 * @param gameState - The game state to export
 * @param name - Name for the exported file
 */
export function exportGameStateToFile(gameState: GameState, name: string): void {
  const scenario = saveCurrentGameState(gameState, { name });
  exportScenarioToFile(scenario);
}

/**
 * Validation result for imported scenarios.
 */
export interface ScenarioValidationResult {
  valid: boolean;
  errors: string[];
  scenario?: LoadableScenario;
}

/**
 * Validate an imported scenario object.
 */
function validateScenario(data: unknown): ScenarioValidationResult {
  const errors: string[] = [];

  if (!data || typeof data !== 'object') {
    return { valid: false, errors: ['Invalid scenario format: not an object'] };
  }

  const obj = data as Record<string, unknown>;

  // Required fields
  if (!obj.id || typeof obj.id !== 'string') {
    errors.push('Missing or invalid "id" field');
  }
  if (!obj.name || typeof obj.name !== 'string') {
    errors.push('Missing or invalid "name" field');
  }
  if (!obj.state || typeof obj.state !== 'object') {
    errors.push('Missing or invalid "state" field');
  }
  if (!obj.boardType || typeof obj.boardType !== 'string') {
    errors.push('Missing or invalid "boardType" field');
  }

  // Validate state structure
  if (obj.state && typeof obj.state === 'object') {
    const state = obj.state as Record<string, unknown>;
    if (!state.board || typeof state.board !== 'object') {
      errors.push('Invalid state: missing "board" field');
    }
    if (!state.players || !Array.isArray(state.players)) {
      errors.push('Invalid state: missing "players" array');
    }
    if (typeof state.currentPlayer !== 'number') {
      errors.push('Invalid state: missing "currentPlayer" field');
    }
    if (!state.currentPhase || typeof state.currentPhase !== 'string') {
      errors.push('Invalid state: missing "currentPhase" field');
    }
  }

  if (errors.length > 0) {
    return { valid: false, errors };
  }

  // Construct validated scenario
  const scenario: LoadableScenario = {
    id: `imported_${Date.now()}_${String(obj.id)}`,
    name: String(obj.name),
    description: String(obj.description || ''),
    category: (obj.category as ScenarioCategory) || 'custom',
    difficulty: obj.difficulty as LoadableScenario['difficulty'],
    tags: Array.isArray(obj.tags) ? obj.tags.map(String) : [],
    boardType: obj.boardType as BoardType,
    playerCount: typeof obj.playerCount === 'number' ? obj.playerCount : 2,
    createdAt: new Date().toISOString(),
    source: 'custom',
    state: obj.state as LoadableScenario['state'],
  };

  return { valid: true, errors: [], scenario };
}

/**
 * Import a scenario from a JSON file.
 *
 * @param file - The uploaded File object
 * @returns Promise resolving to the imported scenario or throwing on error
 */
export async function importScenarioFromFile(file: File): Promise<LoadableScenario> {
  const text = await file.text();

  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch {
    throw new Error('Invalid JSON file');
  }

  const result = validateScenario(data);
  if (!result.valid || !result.scenario) {
    throw new Error(`Invalid scenario file:\n${result.errors.join('\n')}`);
  }

  return result.scenario;
}

/**
 * Import a scenario from a JSON file and save it to localStorage.
 *
 * @param file - The uploaded File object
 * @returns Promise resolving to the saved scenario
 */
export async function importAndSaveScenarioFromFile(file: File): Promise<LoadableScenario> {
  const scenario = await importScenarioFromFile(file);
  saveCustomScenario(scenario);
  return scenario;
}

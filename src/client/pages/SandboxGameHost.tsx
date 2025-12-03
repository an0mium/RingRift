import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { toast } from 'react-hot-toast';
import { BoardView } from '../components/BoardView';
import { ChoiceDialog } from '../components/ChoiceDialog';
import { VictoryModal } from '../components/VictoryModal';
import { GameEventLog } from '../components/GameEventLog';
import { MoveHistory } from '../components/MoveHistory';
import { SandboxTouchControlsPanel } from '../components/SandboxTouchControlsPanel';
import { BoardControlsOverlay } from '../components/BoardControlsOverlay';
import { ScenarioPickerModal } from '../components/ScenarioPickerModal';
import { SelfPlayBrowser } from '../components/SelfPlayBrowser';
import { EvaluationPanel } from '../components/EvaluationPanel';
import type { PositionEvaluationPayload } from '../../shared/types/websocket';
import { SaveStateDialog } from '../components/SaveStateDialog';
import { ReplayPanel } from '../components/ReplayPanel';
import type { LoadableScenario } from '../sandbox/scenarioTypes';
import {
  BoardState,
  BoardType,
  GameState,
  PlayerChoice,
  PlayerChoiceResponseFor,
  Position,
  positionToString,
  CreateGameRequest,
} from '../../shared/types/game';
import { useAuth } from '../contexts/AuthContext';
import { useSandbox, LocalConfig, LocalPlayerType } from '../contexts/SandboxContext';
import { useSandboxInteractions } from '../hooks/useSandboxInteractions';
import { useAutoMoveAnimation } from '../hooks/useMoveAnimation';
import {
  toBoardViewModel,
  toEventLogViewModel,
  toVictoryViewModel,
  deriveBoardDecisionHighlights,
} from '../adapters/gameViewModels';
import { gameApi } from '../services/api';
import { getReplayService } from '../services/ReplayService';
import { storeGameLocally, getPendingCount } from '../services/LocalGameStorage';
import { GameSyncService, type SyncState } from '../services/GameSyncService';
import type { SandboxInteractionHandler } from '../sandbox/ClientSandboxEngine';
import { getGameOverBannerText } from '../utils/gameCopy';
import { serializeGameState } from '../../shared/engine/contracts/serialization';
import { buildTestFixtureFromGameState } from '../sandbox/statePersistence';

const BOARD_PRESETS: Array<{
  value: BoardType;
  label: string;
  subtitle: string;
  blurb: string;
}> = [
  {
    value: 'square8',
    label: '8×8 Compact',
    subtitle: 'Fast tactical battles',
    blurb: 'Ideal for quick tests, fewer territories, emphasizes captures.',
  },
  {
    value: 'square19',
    label: '19×19 Classic',
    subtitle: 'Full RingRift experience',
    blurb: 'All line lengths and ring counts enabled for marathon sessions.',
  },
  {
    value: 'hexagonal',
    label: 'Full Hex',
    subtitle: 'High-mobility frontier',
    blurb: 'Hex adjacency, sweeping captures, and large territory swings.',
  },
];

/** Quick-start scenario presets that configure multiple settings at once */
const QUICK_START_PRESETS: Array<{
  id: string;
  label: string;
  description: string;
  icon: string;
  config: {
    boardType: BoardType;
    numPlayers: number;
    playerTypes: LocalPlayerType[];
  };
}> = [
  {
    id: 'human-vs-ai',
    label: 'Human vs AI',
    description: 'Quick 1v1 against the computer',
    icon: '🤖',
    config: {
      boardType: 'square8',
      numPlayers: 2,
      playerTypes: ['human', 'ai', 'human', 'human'],
    },
  },
  {
    id: 'ai-battle',
    label: 'AI Battle',
    description: 'Watch two AIs compete',
    icon: '⚔️',
    config: {
      boardType: 'square8',
      numPlayers: 2,
      playerTypes: ['ai', 'ai', 'human', 'human'],
    },
  },
  {
    id: 'hotseat',
    label: 'Hotseat',
    description: 'Two humans on one device',
    icon: '👥',
    config: {
      boardType: 'square8',
      numPlayers: 2,
      playerTypes: ['human', 'human', 'human', 'human'],
    },
  },
  {
    id: 'hex-challenge',
    label: 'Hex Challenge',
    description: 'Human vs AI on hex board',
    icon: '⬡',
    config: {
      boardType: 'hexagonal',
      numPlayers: 2,
      playerTypes: ['human', 'ai', 'human', 'human'],
    },
  },
  {
    id: 'four-player',
    label: '4-Player Free-for-All',
    description: 'Chaotic multiplayer on hex',
    icon: '🎲',
    config: {
      boardType: 'hexagonal',
      numPlayers: 4,
      playerTypes: ['human', 'ai', 'ai', 'ai'],
    },
  },
];

const PLAYER_TYPE_META: Record<
  LocalPlayerType,
  { label: string; description: string; accent: string; chip: string }
> = {
  human: {
    label: 'Human',
    description: 'You control every move',
    accent: 'border-emerald-500 text-emerald-200',
    chip: 'bg-emerald-900/40 text-emerald-200',
  },
  ai: {
    label: 'Computer',
    description: 'Local heuristic AI',
    accent: 'border-sky-500 text-sky-200',
    chip: 'bg-sky-900/40 text-sky-200',
  },
};

const PHASE_COPY: Record<
  string,
  {
    label: string;
    summary: string;
  }
> = {
  ring_placement: {
    label: 'Ring Placement',
    summary: 'Place new rings or add to existing stacks while keeping a legal move available.',
  },
  movement: {
    label: 'Movement',
    summary:
      'Pick a stack and move exactly as many spaces as the stack height, respecting blocking rules.',
  },
  // Support both 'capture' and 'chain_capture' phase keys for compatibility
  capture: {
    label: 'Chain Capture',
    summary:
      'Continue capturing until no valid target remains (mandatory per chain capture rules).',
  },
  chain_capture: {
    label: 'Chain Capture',
    summary:
      'Continue capturing until no valid target remains (mandatory per chain capture rules).',
  },
  line_processing: {
    label: 'Line Completion',
    summary: 'Resolve completed lines and make decisions about ring collapses and rewards.',
  },
  territory_processing: {
    label: 'Territory Claim',
    summary:
      'Evaluate disconnected regions, manage territory claims, and apply forced elimination where required.',
  },
};

/**
 * Host component for the local sandbox experience.
 *
 * Responsibilities:
 * - Own sandbox configuration (board type, seats, player kinds) via SandboxContext
 * - Start sandbox games using ClientSandboxEngine, optionally attempting a backend game first
 * - Wire sandbox board interactions and local AI via useSandboxInteractions
 * - Render sandbox-specific HUD (players, selection, phase help, stall diagnostics)
 *
 * Rules semantics remain in the shared TS engine + orchestrator; this host only orchestrates
 * sandbox UI and engine lifecycle.
 */
export const SandboxGameHost: React.FC = () => {
  const navigate = useNavigate();
  const { user } = useAuth();

  const {
    config,
    setConfig,
    isConfigured,
    backendSandboxError,
    setBackendSandboxError,
    sandboxEngine,
    sandboxPendingChoice,
    setSandboxPendingChoice,
    sandboxCaptureChoice,
    setSandboxCaptureChoice,
    sandboxCaptureTargets,
    setSandboxCaptureTargets,
    sandboxLastProgressAt: _sandboxLastProgressAt,
    setSandboxLastProgressAt,
    sandboxStallWarning,
    setSandboxStallWarning,
    sandboxStateVersion: _sandboxStateVersion,
    setSandboxStateVersion,
    initLocalSandboxEngine,
    resetSandboxEngine,
  } = useSandbox();

  const [sandboxEvaluationHistory, setSandboxEvaluationHistory] = useState<
    PositionEvaluationPayload['data'][]
  >([]);
  const [isSandboxAnalysisRunning, setIsSandboxAnalysisRunning] = useState(false);

  // Local-only diagnostics / UX state
  const [isSandboxVictoryModalDismissed, setIsSandboxVictoryModalDismissed] = useState(false);

  // Replay mode state
  const [isInReplayMode, setIsInReplayMode] = useState(false);
  const [replayState, setReplayState] = useState<GameState | null>(null);
  const [replayAnimation, setReplayAnimation] = useState<
    import('../components/BoardView').MoveAnimationData | null
  >(null);

  // Selection + valid target highlighting
  const [selected, setSelected] = useState<Position | undefined>();
  const [validTargets, setValidTargets] = useState<Position[]>([]);
  // Start with the movement grid overlay enabled by default; it helps
  // players understand valid moves and adjacency patterns.
  const [showMovementGrid, setShowMovementGrid] = useState(true);
  const [showValidTargetsOverlay, setShowValidTargetsOverlay] = useState(true);

  // Help / controls overlay for the active sandbox host
  const [showBoardControls, setShowBoardControls] = useState(false);

  // Scenario picker, self-play browser, and save state dialogs
  const [showScenarioPicker, setShowScenarioPicker] = useState(false);
  const [showSelfPlayBrowser, setShowSelfPlayBrowser] = useState(false);
  const [showSaveStateDialog, setShowSaveStateDialog] = useState(false);
  const [lastLoadedScenario, setLastLoadedScenario] = useState<LoadableScenario | null>(null);

  // Game storage state - auto-save completed games to replay database
  const [autoSaveGames, setAutoSaveGames] = useState(true);
  const [gameSaveStatus, setGameSaveStatus] = useState<
    'idle' | 'saving' | 'saved' | 'saved-local' | 'error'
  >('idle');
  const [pendingLocalGames, setPendingLocalGames] = useState(0);
  const [syncState, setSyncState] = useState<SyncState | null>(null);
  const initialGameStateRef = useRef<GameState | null>(null);
  const gameSavedRef = useRef(false);

  const sandboxChoiceResolverRef = useRef<
    ((response: PlayerChoiceResponseFor<PlayerChoice>) => void) | null
  >(null);

  const lastSandboxPhaseRef = useRef<string | null>(null);

  // Sandbox-only visual cue: transient highlight of newly-collapsed line
  // segments, populated from the engine after automatic line processing.
  const [recentLineHighlights, setRecentLineHighlights] = useState<Position[]>([]);

  const {
    handleCellClick: handleSandboxCellClick,
    handleCellDoubleClick: handleSandboxCellDoubleClick,
    handleCellContextMenu: handleSandboxCellContextMenu,
    maybeRunSandboxAiIfNeeded,
    clearSelection: clearSandboxSelection,
  } = useSandboxInteractions({
    selected,
    setSelected,
    validTargets,
    setValidTargets,
    choiceResolverRef: sandboxChoiceResolverRef,
  });

  // Consume any recent line highlights from the sandbox engine whenever the
  // sandbox state version advances. Highlights are cleared automatically
  // after a short delay so they behave as a brief visual cue rather than a
  // persistent overlay.
  useEffect(() => {
    if (!sandboxEngine) {
      setRecentLineHighlights([]);
      return;
    }

    const positions =
      // eslint-disable-next-line @typescript-eslint/no-explicit-any -- duck-typing for optional engine method
      typeof (sandboxEngine as any).consumeRecentLineHighlights === 'function'
        ? sandboxEngine.consumeRecentLineHighlights()
        : [];

    if (positions.length === 0) {
      setRecentLineHighlights([]);
      return;
    }

    setRecentLineHighlights(positions);

    const timeoutId = window.setTimeout(() => {
      setRecentLineHighlights([]);
    }, 1800);

    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [sandboxEngine, _sandboxStateVersion]);

  const requestSandboxEvaluation = useCallback(async () => {
    // Get game state from engine directly to avoid forward reference issues
    const gameState = sandboxEngine?.getGameState();
    if (!sandboxEngine || !gameState) {
      return;
    }

    try {
      setIsSandboxAnalysisRunning(true);

      const serialized = sandboxEngine.getSerializedState();

      const response = await fetch('/api/games/sandbox/evaluate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ state: serialized }),
      });

      if (!response.ok) {
        console.warn('Sandbox evaluation request failed', {
          status: response.status,
        });
        return;
      }

      const data: PositionEvaluationPayload['data'] = await response.json();
      setSandboxEvaluationHistory((prev) => [...prev, data]);
    } catch (err) {
      console.warn('Sandbox evaluation request threw', err);
    } finally {
      setIsSandboxAnalysisRunning(false);
    }
  }, [sandboxEngine]);

  const handleSetupChange = (partial: Partial<LocalConfig>) => {
    setConfig((prev) => {
      const numPlayers = partial.numPlayers;
      return {
        ...prev,
        ...partial,
        playerTypes: numPlayers
          ? prev.playerTypes.map((t, idx) => (idx < numPlayers ? t : prev.playerTypes[idx]))
          : prev.playerTypes,
      };
    });
  };

  const handlePlayerTypeChange = (index: number, type: LocalPlayerType) => {
    setConfig((prev) => {
      const next = [...prev.playerTypes];
      next[index] = type;
      return { ...prev, playerTypes: next };
    });
  };

  const handleQuickStartPreset = (preset: (typeof QUICK_START_PRESETS)[number]) => {
    setConfig((prev) => ({
      ...prev,
      boardType: preset.config.boardType,
      numPlayers: preset.config.numPlayers,
      playerTypes: preset.config.playerTypes,
    }));
  };

  const setAllPlayerTypes = (type: LocalPlayerType) => {
    setConfig((prev) => {
      const next = [...prev.playerTypes];
      for (let i = 0; i < prev.numPlayers; i += 1) {
        next[i] = type;
      }
      return { ...prev, playerTypes: next };
    });
  };

  const createSandboxInteractionHandler = (
    playerTypesSnapshot: LocalPlayerType[]
  ): SandboxInteractionHandler => {
    return {
      async requestChoice<TChoice extends PlayerChoice>(
        choice: TChoice
      ): Promise<PlayerChoiceResponseFor<TChoice>> {
        const playerKind = playerTypesSnapshot[choice.playerNumber - 1] ?? 'human';

        // AI players: pick a random option without involving the UI.
        if (playerKind === 'ai') {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any -- generic choice type narrowing
          const options = (choice as any).options as TChoice['options'];
          const optionsArray = (options as unknown[]) ?? [];
          if (optionsArray.length === 0) {
            throw new Error('SandboxInteractionHandler: no options available for AI choice');
          }
          const selectedOption = optionsArray[
            Math.floor(Math.random() * optionsArray.length)
          ] as TChoice['options'][number];

          return {
            choiceId: choice.id,
            playerNumber: choice.playerNumber,
            choiceType: choice.type,
            selectedOption,
          } as PlayerChoiceResponseFor<TChoice>;
        }

        // Human players
        if (choice.type === 'capture_direction') {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any -- capture_direction type narrowing
          const anyChoice = choice as any;
          const options = (anyChoice.options ?? []) as Array<{ landingPosition: Position }>;
          const targets: Position[] = options.map((opt) => opt.landingPosition);
          setSandboxCaptureChoice(choice);
          setSandboxCaptureTargets(targets);
        } else {
          setSandboxPendingChoice(choice);
        }

        return new Promise<PlayerChoiceResponseFor<TChoice>>((resolve) => {
          sandboxChoiceResolverRef.current = ((response: PlayerChoiceResponseFor<PlayerChoice>) => {
            resolve(response as PlayerChoiceResponseFor<TChoice>);
          }) as (response: PlayerChoiceResponseFor<PlayerChoice>) => void;
        });
      },
    };
  };

  /**
   * Load a scenario from the scenario picker.
   * This initializes the sandbox engine from a pre-existing serialized game state.
   */
  const handleLoadScenario = (scenario: LoadableScenario) => {
    // Update config to match scenario settings
    setConfig((prev) => ({
      ...prev,
      boardType: scenario.boardType,
      numPlayers: scenario.playerCount,
    }));

    // Get player types (default to human vs AI for 2 players)
    const playerTypes: LocalPlayerType[] =
      scenario.playerCount === 2
        ? ['human', 'ai', 'human', 'human']
        : config.playerTypes.slice(0, scenario.playerCount);

    // Create interaction handler
    const interactionHandler = createSandboxInteractionHandler(playerTypes);

    // Initialize sandbox engine with the scenario state
    const engine = initLocalSandboxEngine({
      boardType: scenario.boardType,
      numPlayers: scenario.playerCount,
      playerTypes,
      interactionHandler,
    });

    // Load the serialized state into the engine
    engine.initFromSerializedState(scenario.state, playerTypes, interactionHandler);

    // Reset UI state
    setSelected(undefined);
    setValidTargets([]);
    setSandboxPendingChoice(null);
    setIsSandboxVictoryModalDismissed(false);
    setBackendSandboxError(null);
    setSandboxCaptureChoice(null);
    setSandboxCaptureTargets([]);
    setSandboxStallWarning(null);
    setSandboxLastProgressAt(null);
    setSandboxStateVersion(0);
    setLastLoadedScenario(scenario);

    toast.success(`Loaded scenario: ${scenario.name}`);
  };

  /**
   * Fork from a replay position - loads the replay state into the sandbox engine
   * as a new playable game.
   */
  const handleForkFromReplay = (state: GameState) => {
    // Update config to match the replay state
    const numPlayers = state.players.length;
    setConfig((prev) => ({
      ...prev,
      boardType: state.board.type,
      numPlayers,
    }));

    // Default player types for forked game
    const playerTypes: LocalPlayerType[] = state.players.map((_, idx) =>
      idx === 0 ? 'human' : 'ai'
    );

    // Create interaction handler
    const interactionHandler = createSandboxInteractionHandler(playerTypes);

    // Initialize sandbox engine
    const engine = initLocalSandboxEngine({
      boardType: state.board.type,
      numPlayers,
      playerTypes,
      interactionHandler,
    });

    // Load the state into the engine (convert GameState to SerializedGameState)
    const serialized = serializeGameState(state);
    engine.initFromSerializedState(serialized, playerTypes, interactionHandler);

    // Reset UI state
    setSelected(undefined);
    setValidTargets([]);
    setSandboxPendingChoice(null);
    setIsSandboxVictoryModalDismissed(false);
    setBackendSandboxError(null);
    setSandboxCaptureChoice(null);
    setSandboxCaptureTargets([]);
    setSandboxStallWarning(null);
    setSandboxLastProgressAt(null);
    setSandboxStateVersion(0);
    setLastLoadedScenario(null);

    // Exit replay mode
    setIsInReplayMode(false);
    setReplayState(null);

    toast.success('Forked from replay position');
  };

  const handleResetScenario = () => {
    if (!lastLoadedScenario) {
      return;
    }

    const scenario = lastLoadedScenario;

    // Mirror the logic from handleLoadScenario so reset behaves the same as a
    // fresh scenario load.
    setConfig((prev) => ({
      ...prev,
      boardType: scenario.boardType,
      numPlayers: scenario.playerCount,
    }));

    const playerTypes: LocalPlayerType[] =
      scenario.playerCount === 2
        ? ['human', 'ai', 'human', 'human']
        : config.playerTypes.slice(0, scenario.playerCount);

    const interactionHandler = createSandboxInteractionHandler(playerTypes);

    const engine = initLocalSandboxEngine({
      boardType: scenario.boardType,
      numPlayers: scenario.playerCount,
      playerTypes,
      interactionHandler,
    });

    engine.initFromSerializedState(scenario.state, playerTypes, interactionHandler);

    setSelected(undefined);
    setValidTargets([]);
    setSandboxPendingChoice(null);
    setSandboxCaptureChoice(null);
    setSandboxCaptureTargets([]);
    setSandboxStallWarning(null);
    setSandboxLastProgressAt(null);
    setIsSandboxVictoryModalDismissed(false);
    setBackendSandboxError(null);
    setSandboxStateVersion((v) => v + 1);

    toast.success(`Scenario reset: ${scenario.name}`);
  };

  const handleStartLocalGame = async () => {
    const nextBoardType = config.boardType;

    // First, attempt to create a real backend game using the same CreateGameRequest
    // shape as the lobby. On success, navigate into the real backend game route.
    try {
      const payload: CreateGameRequest = {
        boardType: nextBoardType,
        maxPlayers: config.numPlayers,
        isRated: false,
        isPrivate: true,
        timeControl: {
          type: 'rapid',
          initialTime: 600,
          increment: 0,
        },
        aiOpponents: (() => {
          const aiSeats = config.playerTypes
            .slice(0, config.numPlayers)
            .filter((t) => t === 'ai').length;
          if (aiSeats <= 0) return undefined;
          return {
            count: aiSeats,
            difficulty: Array(aiSeats).fill(5),
            mode: 'service',
            aiType: 'heuristic',
          };
        })(),
        // Mirror lobby behaviour: default-enable the pie rule for 2-player
        // backend sandbox games. Local-only sandbox games (fallback path)
        // continue to use the shared engine's defaults.
        rulesOptions: config.numPlayers === 2 ? { swapRuleEnabled: true } : undefined,
      };

      const game = await gameApi.createGame(payload);
      navigate(`/game/${game.id}`);
      return;
    } catch (err) {
      console.error('Failed to create backend sandbox game, falling back to local-only board', err);
      setBackendSandboxError(
        'Backend sandbox game could not be created; falling back to local-only board only.'
      );
    }

    // Fallback: local sandbox engine using orchestrator-first semantics.
    const interactionHandler = createSandboxInteractionHandler(
      config.playerTypes.slice(0, config.numPlayers)
    );
    const engine = initLocalSandboxEngine({
      boardType: nextBoardType,
      numPlayers: config.numPlayers,
      playerTypes: config.playerTypes.slice(0, config.numPlayers) as LocalPlayerType[],
      interactionHandler,
    });

    setSelected(undefined);
    setValidTargets([]);
    setSandboxPendingChoice(null);

    // If the first player is an AI, immediately start the sandbox AI turn loop.
    if (engine) {
      const state = engine.getGameState();
      const current = state.players.find((p) => p.playerNumber === state.currentPlayer);
      if (current && current.type === 'ai') {
        maybeRunSandboxAiIfNeeded();
      }
    }
  };

  const handleCopySandboxTrace = async () => {
    try {
      if (typeof window === 'undefined') {
        return;
      }

      // eslint-disable-next-line @typescript-eslint/no-explicit-any -- accessing debug global
      const anyWindow = window as any;
      const trace = anyWindow.__RINGRIFT_SANDBOX_TRACE__ ?? [];
      const payload = JSON.stringify(trace, null, 2);

      if (
        typeof navigator !== 'undefined' &&
        navigator.clipboard &&
        navigator.clipboard.writeText
      ) {
        await navigator.clipboard.writeText(payload);
        toast.success('Sandbox AI trace copied to clipboard');
      } else {
        // eslint-disable-next-line no-console
        console.log('Sandbox AI trace', trace);
        toast.success('Sandbox AI trace logged to console (clipboard API unavailable).');
      }
    } catch (err) {
      console.error('Failed to export sandbox AI trace', err);
      toast.error('Failed to export sandbox AI trace; see console for details.');
    }
  };

  const handleCopySandboxFixture = async () => {
    try {
      if (!sandboxGameState) {
        toast.error('No sandbox game is currently active.');
        return;
      }

      const fixture = buildTestFixtureFromGameState(sandboxGameState);
      const payload = JSON.stringify(fixture, null, 2);

      if (
        typeof navigator !== 'undefined' &&
        navigator.clipboard &&
        navigator.clipboard.writeText
      ) {
        await navigator.clipboard.writeText(payload);
        toast.success('Sandbox test fixture copied to clipboard');
      } else {
        // eslint-disable-next-line no-console
        console.log('Sandbox test fixture', fixture);
        toast.success('Sandbox test fixture logged to console (clipboard API unavailable).');
      }
    } catch (err) {
      console.error('Failed to export sandbox test fixture', err);
      toast.error('Failed to export sandbox test fixture; see console for details.');
    }
  };

  // Game view once configured (local sandbox)
  const sandboxGameState: GameState | null = sandboxEngine ? sandboxEngine.getGameState() : null;
  const sandboxVictoryResult = sandboxEngine ? sandboxEngine.getVictoryResult() : null;

  // When in replay mode, show the replay state instead of the sandbox state
  const displayGameState = isInReplayMode && replayState ? replayState : sandboxGameState;
  const sandboxBoardState: BoardState | null = displayGameState?.board ?? null;

  // Move animations - auto-detects moves from game state changes
  const { pendingAnimation, clearAnimation } = useAutoMoveAnimation(sandboxGameState);

  const sandboxGameOverBannerText =
    sandboxVictoryResult && isSandboxVictoryModalDismissed && sandboxVictoryResult.reason
      ? getGameOverBannerText(sandboxVictoryResult.reason)
      : null;

  const boardTypeValue = sandboxBoardState?.type ?? config.boardType;
  const boardPresetInfo = BOARD_PRESETS.find((preset) => preset.value === boardTypeValue);
  const boardDisplayLabel = boardPresetInfo?.label ?? boardTypeValue;
  const boardDisplaySubtitle = boardPresetInfo?.subtitle ?? 'Custom configuration';
  const _boardDisplayBlurb =
    boardPresetInfo?.blurb ?? 'Custom layout selected for this local sandbox match.';

  // When a ring elimination decision is active in the sandbox, repurpose the
  // heuristic/status chip under the board as an explicit elimination prompt so
  // it mirrors the backend HUD directive.
  const isRingEliminationChoice =
    (sandboxCaptureChoice ?? sandboxPendingChoice)?.type === 'ring_elimination';

  // When in chain_capture with available continuation segments, surface an
  // attention-style chip prompting the user to continue the chain. This mirrors
  // backend HUD semantics for mandatory chain continuation.
  const isChainCaptureContinuationStep = !!(
    sandboxGameState &&
    sandboxGameState.currentPhase === 'chain_capture' &&
    sandboxEngine &&
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- duck-typing for optional engine method
    typeof (sandboxEngine as any).getValidMoves === 'function' &&
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- accessing internal engine method
    (sandboxEngine as any)
      .getValidMoves(sandboxGameState.currentPlayer)
      .some((m: { type: string }) => m.type === 'continue_capture_segment')
  );

  const sandboxPlayersList =
    sandboxGameState?.players ??
    Array.from({ length: config.numPlayers }, (_, idx) => ({
      playerNumber: idx + 1,
      username: `Player ${idx + 1}`,
      type: config.playerTypes[idx] ?? 'human',
      ringsInHand: 0,
      eliminatedRings: 0,
      territorySpaces: 0,
    }));

  const sandboxCurrentPlayerNumber = sandboxGameState?.currentPlayer ?? 1;
  const _sandboxCurrentPlayer =
    sandboxPlayersList.find((p) => p.playerNumber === sandboxCurrentPlayerNumber) ??
    sandboxPlayersList[0];

  const sandboxPhaseKey = sandboxGameState?.currentPhase ?? 'ring_placement';
  const sandboxPhaseDetails = PHASE_COPY[sandboxPhaseKey] ?? PHASE_COPY.ring_placement;

  useEffect(() => {
    if (!sandboxGameState) {
      lastSandboxPhaseRef.current = null;
      return;
    }

    const previousPhase = lastSandboxPhaseRef.current;
    const nextPhase = sandboxGameState.currentPhase;

    if (previousPhase !== nextPhase) {
      const stacksSnapshot = Array.from(sandboxGameState.board.stacks.entries()).map(
        ([key, stack]) => ({
          key,
          height: stack.stackHeight,
          cap: stack.capHeight,
          controllingPlayer: stack.controllingPlayer,
        })
      );

      // eslint-disable-next-line no-console
      console.log('[SandboxPhaseDebug][SandboxGameHost] Phase change in sandbox', {
        from: previousPhase,
        to: nextPhase,
        currentPlayer: sandboxGameState.currentPlayer,
        gameStatus: sandboxGameState.gameStatus,
        stacks: stacksSnapshot,
      });

      if (nextPhase === 'line_processing') {
        const formedLinesSnapshot =
          sandboxGameState.board.formedLines?.map((line) => ({
            player: line.player,
            length: line.length,
            positions: line.positions.map((pos) => positionToString(pos)),
          })) ?? [];

        // eslint-disable-next-line no-console
        console.log(
          '[SandboxPhaseDebug][SandboxGameHost] Entered line_processing with formedLines',
          formedLinesSnapshot
        );
      }

      lastSandboxPhaseRef.current = nextPhase;
    }
  }, [sandboxGameState]);

  const humanSeatCount = sandboxPlayersList.filter((p) => p.type === 'human').length;
  const aiSeatCount = sandboxPlayersList.length - humanSeatCount;

  // Whenever the sandbox state reflects an active AI turn, trigger the
  // sandbox AI loop after a short delay. This keeps AI progression in
  // sync with orchestrator-driven state changes (including line/territory
  // processing and elimination decisions) without requiring an extra
  // board click from the user.
  useEffect(() => {
    if (!sandboxEngine || !sandboxGameState) {
      return;
    }

    const current = sandboxGameState.players.find(
      (p) => p.playerNumber === sandboxGameState.currentPlayer
    );

    if (sandboxGameState.gameStatus !== 'active' || !current || current.type !== 'ai') {
      return;
    }

    setSandboxLastProgressAt(Date.now());
    setSandboxStallWarning(null);

    const timeoutId = window.setTimeout(() => {
      maybeRunSandboxAiIfNeeded();
    }, 60);

    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [
    sandboxEngine,
    sandboxGameState,
    maybeRunSandboxAiIfNeeded,
    setSandboxLastProgressAt,
    setSandboxStallWarning,
  ]);

  // Derive board VM + HUD-like summaries
  const primaryValidTargets =
    sandboxCaptureTargets.length > 0 ? sandboxCaptureTargets : validTargets;

  const displayedValidTargets = showValidTargetsOverlay ? primaryValidTargets : [];

  // Derive decision-phase highlights from the current sandbox GameState and
  // whichever PlayerChoice is currently active. Capture-direction choices
  // take precedence over generic pending choices so that landing/target
  // geometry is always visible while the capture UI is open.
  const activePendingChoice: PlayerChoice | null = sandboxCaptureChoice ?? sandboxPendingChoice;

  const baseDecisionHighlights =
    sandboxGameState && activePendingChoice
      ? deriveBoardDecisionHighlights(sandboxGameState, activePendingChoice)
      : undefined;

  // Merge transient line highlights into the decision highlight model so
  // recently-collapsed lines receive a brief visual cue even when no
  // explicit line-order/reward choice is surfaced.
  let decisionHighlights = baseDecisionHighlights;
  if (recentLineHighlights.length > 0) {
    const recentKeys = new Set(recentLineHighlights.map((pos) => positionToString(pos)));
    const existing = baseDecisionHighlights?.highlights ?? [];

    const extraHighlights = Array.from(recentKeys)
      .filter((key) => !existing.some((h) => h.positionKey === key))
      .map((key) => ({
        positionKey: key,
        intensity: 'primary' as const,
      }));

    if (extraHighlights.length > 0) {
      decisionHighlights = {
        choiceKind: baseDecisionHighlights?.choiceKind ?? 'line_order',
        highlights: [...existing, ...extraHighlights],
      };
    }
  }

  const sandboxBoardViewModel = sandboxBoardState
    ? toBoardViewModel(sandboxBoardState, {
        selectedPosition: selected,
        validTargets: displayedValidTargets,
        decisionHighlights,
      })
    : null;

  useEffect(() => {
    if (!sandboxBoardState || !sandboxBoardViewModel) {
      return;
    }

    if (sandboxPhaseKey !== 'line_processing') {
      return;
    }

    const stacksSnapshot = Array.from(sandboxBoardState.stacks.entries()).map(([key, stack]) => ({
      key,
      height: stack.stackHeight,
      cap: stack.capHeight,
      controllingPlayer: stack.controllingPlayer,
    }));

    const decisionHighlightsSnapshot =
      decisionHighlights?.highlights?.map((h) => h.positionKey) ?? [];

    // eslint-disable-next-line no-console
    console.log('[SandboxPhaseDebug][SandboxGameHost] BoardView props in line_processing', {
      boardType: sandboxBoardState.type,
      phase: sandboxPhaseKey,
      stacks: stacksSnapshot,
      selectedPosition: selected ? positionToString(selected) : null,
      validTargets: displayedValidTargets.map((pos) => positionToString(pos)),
      decisionHighlights: decisionHighlightsSnapshot,
    });
  }, [
    sandboxBoardState,
    sandboxBoardViewModel,
    sandboxPhaseKey,
    selected,
    displayedValidTargets,
    decisionHighlights,
  ]);

  const sandboxVictoryViewModel = sandboxVictoryResult
    ? toVictoryViewModel(
        sandboxVictoryResult,
        sandboxGameState?.players ?? [],
        sandboxGameState ?? undefined,
        {
          currentUserId: user?.id,
          isDismissed: isSandboxVictoryModalDismissed,
        }
      )
    : null;

  const sandboxHudPlayers = sandboxPlayersList.map((player) => ({
    playerNumber: player.playerNumber,
    username: player.username || `Player ${player.playerNumber}`,
    type: player.type,
    ringsInHand: player.ringsInHand,
    eliminatedRings: player.eliminatedRings,
    territorySpaces: player.territorySpaces,
    isCurrent: player.playerNumber === sandboxCurrentPlayerNumber,
  }));

  const sandboxModeNotes = [
    `Board: ${boardDisplayLabel}`,
    `${humanSeatCount} human seat${humanSeatCount === 1 ? '' : 's'} · ${aiSeatCount} AI`,
    sandboxEngine
      ? 'Engine parity mode with local AI and choice handler.'
      : 'Legacy local sandbox fallback (no backend).',
    'Runs entirely in-browser; use "Change Setup" to switch configurations.',
  ];

  const sandboxHudViewModel = {
    players: sandboxHudPlayers,
    phaseDetails: sandboxPhaseDetails,
    modeNotes: sandboxModeNotes,
  };

  const sandboxEventLogViewModel = toEventLogViewModel(
    sandboxGameState?.history ?? [],
    [],
    sandboxVictoryResult,
    { maxEntries: 40 }
  );

  const selectedStackDetails = (() => {
    if (!sandboxBoardState || !selected) return null;
    const key = positionToString(selected);
    const stack = sandboxBoardState.stacks.get(key);
    if (!stack) return null;
    return {
      height: stack.stackHeight,
      cap: stack.capHeight,
      controllingPlayer: stack.controllingPlayer,
    };
  })();

  const activePlayerTypes = config.playerTypes.slice(0, config.numPlayers);
  const setupHumanSeatCount = activePlayerTypes.filter((t) => t === 'human').length;
  const setupAiSeatCount = activePlayerTypes.length - setupHumanSeatCount;
  const selectedBoardPreset =
    BOARD_PRESETS.find((preset) => preset.value === config.boardType) ?? BOARD_PRESETS[0];

  // Keyboard shortcuts for sandbox overlay:
  // - "?" (Shift + "/") toggles the Board Controls overlay when a sandbox game is active.
  // - "Escape" closes the overlay when open.
  useEffect(() => {
    if (!isConfigured || !sandboxEngine) {
      return;
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented) return;

      const target = event.target as HTMLElement | null;
      if (target) {
        const tagName = target.tagName;
        const isEditableTag = tagName === 'INPUT' || tagName === 'TEXTAREA' || tagName === 'SELECT';
        const isContentEditable = target.isContentEditable;
        if (isEditableTag || isContentEditable) {
          return;
        }
      }

      if (event.key === '?' || (event.key === '/' && event.shiftKey)) {
        event.preventDefault();
        setShowBoardControls((prev) => !prev);
        return;
      }

      if (event.key === 'Escape' && showBoardControls) {
        event.preventDefault();
        setShowBoardControls(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [isConfigured, sandboxEngine, showBoardControls]);

  // Capture initial game state when engine is created for game storage
  useEffect(() => {
    if (!sandboxEngine) {
      // Reset refs when engine is destroyed
      initialGameStateRef.current = null;
      gameSavedRef.current = false;
      setGameSaveStatus('idle');
      return;
    }
    // Capture initial state only once per game (when moveHistory is empty)
    const currentState = sandboxEngine.getGameState();
    if (currentState.moveHistory.length === 0 && !initialGameStateRef.current) {
      initialGameStateRef.current = structuredClone(currentState);
      gameSavedRef.current = false;
      setGameSaveStatus('idle');
    }
  }, [sandboxEngine]);

  // Start game sync service and subscribe to state updates
  useEffect(() => {
    GameSyncService.start();
    const unsubscribe = GameSyncService.subscribe((state) => {
      setSyncState(state);
      setPendingLocalGames(state.pendingCount);
    });
    return () => {
      unsubscribe();
      GameSyncService.stop();
    };
  }, []);

  // Auto-save completed games to replay database when victory is detected
  useEffect(() => {
    if (!autoSaveGames || !sandboxVictoryResult || gameSavedRef.current) {
      return;
    }

    const saveCompletedGame = async () => {
      const finalState = sandboxEngine?.getGameState();
      const initialState = initialGameStateRef.current;

      if (!finalState || !initialState) {
        console.warn('[SandboxGameHost] Cannot save game: missing state');
        return;
      }

      const metadata = {
        source: 'sandbox',
        boardType: finalState.board.type,
        numPlayers: finalState.players.length,
        playerTypes: config.playerTypes.slice(0, config.numPlayers),
        victoryReason: sandboxVictoryResult.reason,
        winnerPlayerNumber: sandboxVictoryResult.winner,
      };

      try {
        setGameSaveStatus('saving');
        const replayService = getReplayService();
        const result = await replayService.storeGame({
          initialState,
          finalState,
          moves: finalState.moveHistory as unknown as Record<string, unknown>[],
          metadata,
        });

        if (result.success) {
          gameSavedRef.current = true;
          setGameSaveStatus('saved');
          toast.success(`Game saved (${result.totalMoves} moves)`);
        } else {
          // Server rejected - try local fallback
          throw new Error('Server rejected game storage');
        }
      } catch (error) {
        console.warn('[SandboxGameHost] Server save failed, trying local storage:', error);

        // Fallback to IndexedDB local storage
        try {
          const localResult = await storeGameLocally(
            initialState,
            finalState,
            finalState.moveHistory as unknown[],
            metadata
          );

          if (localResult.success) {
            gameSavedRef.current = true;
            setGameSaveStatus('saved-local');
            const newCount = await getPendingCount();
            setPendingLocalGames(newCount);
            toast.success('Game saved locally (will sync when server available)', {
              icon: '💾',
            });
          } else {
            setGameSaveStatus('error');
            toast.error('Failed to save game');
          }
        } catch (localError) {
          console.error('[SandboxGameHost] Local storage also failed:', localError);
          setGameSaveStatus('error');
          toast.error('Failed to save game (storage unavailable)');
        }
      }
    };

    saveCompletedGame();
  }, [autoSaveGames, sandboxVictoryResult, sandboxEngine, config.playerTypes, config.numPlayers]);

  // Pre-game setup view
  if (!isConfigured || !sandboxEngine) {
    return (
      <div className="min-h-screen bg-slate-950 text-slate-100">
        <div className="container mx-auto px-4 py-8 space-y-6">
          <header>
            <h1 className="text-3xl font-bold mb-1">Start a RingRift Game (Local Sandbox)</h1>
            <p className="text-sm text-slate-400">
              This mode runs entirely in the browser using a local board. To view or play a real
              server-backed game, navigate to a URL with a game ID (e.g.
              <code className="ml-1 text-xs text-slate-300">/game/:gameId</code>).
            </p>
          </header>

          {/* Quick-start presets */}
          <section className="p-4 rounded-2xl bg-slate-800/50 border border-slate-700">
            <div className="flex items-center justify-between mb-3">
              <div>
                <p className="text-xs uppercase tracking-wide text-slate-400">Quick Start</p>
                <h2 className="text-lg font-semibold text-white">Choose a preset</h2>
              </div>
            </div>
            <div className="flex flex-wrap gap-2">
              {QUICK_START_PRESETS.map((preset) => (
                <button
                  key={preset.id}
                  type="button"
                  onClick={() => handleQuickStartPreset(preset)}
                  className="flex items-center gap-2 px-3 py-2 rounded-xl border border-slate-600 bg-slate-900/60 text-slate-200 hover:border-emerald-400 hover:text-emerald-200 transition text-sm"
                >
                  <span className="text-lg" role="img" aria-hidden="true">
                    {preset.icon}
                  </span>
                  <div className="text-left">
                    <p className="font-semibold">{preset.label}</p>
                    <p className="text-xs text-slate-400">{preset.description}</p>
                  </div>
                </button>
              ))}
            </div>
          </section>

          {/* Load Scenario section */}
          <section className="p-4 rounded-2xl bg-slate-800/50 border border-slate-700">
            <div className="flex items-center justify-between mb-3">
              <div>
                <p className="text-xs uppercase tracking-wide text-slate-400">Scenarios</p>
                <h2 className="text-lg font-semibold text-white">Load a saved scenario</h2>
              </div>
            </div>
            <p className="text-sm text-slate-400 mb-3">
              Load test vectors, curated learning scenarios, or your own saved game states.
            </p>
            <button
              type="button"
              onClick={() => setShowScenarioPicker(true)}
              className="px-4 py-2 rounded-xl border border-slate-600 bg-slate-900/60 text-slate-200 hover:border-emerald-400 hover:text-emerald-200 transition text-sm font-medium"
            >
              Browse Scenarios
            </button>
          </section>

          {/* Self-Play Games section */}
          <section className="p-4 rounded-2xl bg-slate-800/50 border border-slate-700">
            <div className="flex items-center justify-between mb-3">
              <div>
                <p className="text-xs uppercase tracking-wide text-slate-400">AI Training</p>
                <h2 className="text-lg font-semibold text-white">Browse self-play games</h2>
              </div>
            </div>
            <p className="text-sm text-slate-400 mb-3">
              Load and replay games recorded during CMA-ES training, self-play soaks, and other AI
              training activities.
            </p>
            <button
              type="button"
              onClick={() => setShowSelfPlayBrowser(true)}
              className="px-4 py-2 rounded-xl border border-slate-600 bg-slate-900/60 text-slate-200 hover:border-sky-400 hover:text-sky-200 transition text-sm font-medium"
            >
              Browse Self-Play Games
            </button>
          </section>

          <section className="grid gap-6 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
            <div className="p-5 rounded-2xl bg-slate-800/70 border border-slate-700 space-y-6 text-slate-100 shadow-lg">
              {backendSandboxError && (
                <div className="p-3 text-sm text-red-300 bg-red-900/40 border border-red-700 rounded-lg">
                  {backendSandboxError}
                </div>
              )}

              <div className="space-y-3">
                <div className="flex items-center justify-between flex-wrap gap-2">
                  <div>
                    <p className="text-xs uppercase tracking-wide text-slate-400">Players</p>
                    <h2 className="text-lg font-semibold text-white">Seats & control</h2>
                  </div>
                  <div className="flex gap-2 text-xs">
                    {[2, 3, 4].map((count) => (
                      <button
                        key={count}
                        type="button"
                        onClick={() => handleSetupChange({ numPlayers: count })}
                        className={`px-2 py-1 rounded-full border ${
                          config.numPlayers === count
                            ? 'border-emerald-400 text-emerald-200 bg-emerald-900/30'
                            : 'border-slate-600 text-slate-300 hover:border-slate-400'
                        }`}
                      >
                        {count} Players
                      </button>
                    ))}
                  </div>
                </div>

                <div className="space-y-3">
                  {Array.from({ length: config.numPlayers }, (_, i) => {
                    const type = config.playerTypes[i];
                    const meta = PLAYER_TYPE_META[type];
                    return (
                      <div
                        key={i}
                        className={`rounded-xl border bg-slate-900/60 px-4 py-3 flex items-center justify-between gap-4 ${meta.accent}`}
                      >
                        <div>
                          <p className="text-sm font-semibold text-white">Player {i + 1}</p>
                          <p className="text-xs text-slate-300">{meta.description}</p>
                        </div>
                        <div className="flex gap-2">
                          {(['human', 'ai'] as LocalPlayerType[]).map((candidate) => {
                            const isActive = type === candidate;
                            return (
                              <button
                                key={candidate}
                                type="button"
                                onClick={() => handlePlayerTypeChange(i, candidate)}
                                className={`px-3 py-1 rounded-full border text-xs font-semibold transition ${
                                  isActive
                                    ? 'border-white/80 text-white bg-white/10'
                                    : 'border-slate-600 text-slate-300 hover:border-slate-400'
                                }`}
                              >
                                {PLAYER_TYPE_META[candidate].label}
                              </button>
                            );
                          })}
                        </div>
                      </div>
                    );
                  })}
                </div>

                <div className="flex flex-wrap gap-2 text-xs">
                  <button
                    type="button"
                    onClick={() => setAllPlayerTypes('human')}
                    className="px-3 py-1 rounded-full border border-slate-500 text-slate-200 hover:border-emerald-400 hover:text-emerald-200 transition"
                  >
                    All Human
                  </button>
                  <button
                    type="button"
                    onClick={() => setAllPlayerTypes('ai')}
                    className="px-3 py-1 rounded-full border border-slate-500 text-slate-200 hover:border-sky-400 hover:text-sky-200 transition"
                  >
                    All AI
                  </button>
                </div>
              </div>

              <div className="space-y-3">
                <div className="flex items-center justify-between flex-wrap gap-2">
                  <div>
                    <p className="text-xs uppercase tracking-wide text-slate-400">Board</p>
                    <h2 className="text-lg font-semibold text-white">Choose a layout</h2>
                  </div>
                </div>

                <div className="grid gap-3 sm:grid-cols-2">
                  {BOARD_PRESETS.map((preset) => {
                    const isSelected = preset.value === config.boardType;
                    return (
                      <button
                        key={preset.value}
                        type="button"
                        onClick={() => handleSetupChange({ boardType: preset.value })}
                        className={`p-4 text-left rounded-2xl border transition shadow-sm ${
                          isSelected
                            ? 'border-emerald-400 bg-emerald-900/20 text-white'
                            : 'border-slate-600 bg-slate-900/60 text-slate-200 hover:border-slate-400'
                        }`}
                      >
                        <span className="text-xs uppercase tracking-wide text-slate-400">
                          {preset.subtitle}
                        </span>
                        <p className="text-lg font-semibold">{preset.label}</p>
                        <p className="text-xs text-slate-300 mt-1">{preset.blurb}</p>
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>

            <div className="p-5 rounded-2xl bg-slate-900/70 border border-slate-700 text-slate-100 shadow-lg space-y-4">
              <div>
                <p className="text-xs uppercase tracking-wide text-slate-400">Summary</p>
                <h2 className="text-xl font-bold text-white">{selectedBoardPreset.label}</h2>
                <p className="text-sm text-slate-300">{selectedBoardPreset.blurb}</p>
              </div>

              <div className="space-y-3 text-sm">
                <div className="flex items-center justify-between">
                  <span className="text-slate-300">Humans</span>
                  <span className="font-semibold">{setupHumanSeatCount}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-300">AI opponents</span>
                  <span className="font-semibold">{setupAiSeatCount}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-300">Total seats</span>
                  <span className="font-semibold">{config.numPlayers}</span>
                </div>
              </div>

              <div className="space-y-2">
                <p className="text-xs text-slate-400">
                  We first attempt to stand up a backend game with these settings. If that fails, we
                  fall back to a purely client-local sandbox so you can still test moves offline.
                </p>
                <button
                  type="button"
                  onClick={handleStartLocalGame}
                  className="w-full px-4 py-3 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-sm font-semibold text-white shadow-lg shadow-emerald-900/40 transition"
                >
                  Launch Game
                </button>
              </div>
            </div>
          </section>
        </div>

        <ScenarioPickerModal
          isOpen={showScenarioPicker}
          onClose={() => setShowScenarioPicker(false)}
          onSelectScenario={handleLoadScenario}
        />

        <SelfPlayBrowser
          isOpen={showSelfPlayBrowser}
          onClose={() => setShowSelfPlayBrowser(false)}
          onSelectGame={handleLoadScenario}
        />
      </div>
    );
  }

  // === Active sandbox game ===
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="container mx-auto px-4 py-8 space-y-4">
        {sandboxStallWarning && (
          <div className="p-3 rounded-xl border border-amber-500/70 bg-amber-900/40 text-amber-100 text-xs flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
            <span>{sandboxStallWarning}</span>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={handleCopySandboxTrace}
                className="px-3 py-1 rounded-lg border border-amber-300 bg-amber-800/70 text-[11px] font-semibold hover:border-amber-100 hover:bg-amber-700/80"
              >
                Copy AI trace
              </button>
              <button
                type="button"
                onClick={() => setSandboxStallWarning(null)}
                className="px-2 py-1 rounded-lg border border-slate-500 text-[11px] hover:border-slate-300"
              >
                Dismiss
              </button>
            </div>
          </div>
        )}

        {sandboxGameOverBannerText && (
          <div className="p-3 rounded-xl border border-emerald-500/70 bg-emerald-900/40 text-emerald-100 text-xs">
            {sandboxGameOverBannerText}
          </div>
        )}

        {sandboxGameState && (
          <VictoryModal
            isOpen={!!sandboxVictoryResult && !isSandboxVictoryModalDismissed}
            viewModel={sandboxVictoryViewModel}
            onClose={() => {
              setIsSandboxVictoryModalDismissed(true);
            }}
            onReturnToLobby={() => {
              resetSandboxEngine();
              setSelected(undefined);
              setValidTargets([]);
              setBackendSandboxError(null);
              setSandboxPendingChoice(null);
              setIsSandboxVictoryModalDismissed(false);
            }}
            onRematch={() => {
              // Reset state and start a new game with the same configuration
              setIsSandboxVictoryModalDismissed(false);
              setSelected(undefined);
              setValidTargets([]);
              setSandboxPendingChoice(null);
              setSandboxCaptureChoice(null);
              setSandboxCaptureTargets([]);
              setSandboxStallWarning(null);
              setSandboxLastProgressAt(null);

              // Re-initialize with the same config
              const interactionHandler = createSandboxInteractionHandler(
                config.playerTypes.slice(0, config.numPlayers)
              );
              const engine = initLocalSandboxEngine({
                boardType: config.boardType,
                numPlayers: config.numPlayers,
                playerTypes: config.playerTypes.slice(0, config.numPlayers) as LocalPlayerType[],
                interactionHandler,
              });

              // If the first player is AI, start the AI turn loop
              if (engine) {
                const state = engine.getGameState();
                const current = state.players.find((p) => p.playerNumber === state.currentPlayer);
                if (current && current.type === 'ai') {
                  maybeRunSandboxAiIfNeeded();
                }
              }

              toast.success('New game started with the same settings!');
            }}
          />
        )}

        <ChoiceDialog
          choice={
            sandboxPendingChoice && sandboxPendingChoice.type === 'ring_elimination'
              ? null
              : sandboxPendingChoice
          }
          deadline={null}
          onSelectOption={(choice, option) => {
            const resolver = sandboxChoiceResolverRef.current;
            if (resolver) {
              resolver({
                choiceId: choice.id,
                playerNumber: choice.playerNumber,
                choiceType: choice.type,
                selectedOption: option,
              } as PlayerChoiceResponseFor<PlayerChoice>);
              sandboxChoiceResolverRef.current = null;
            }
            setSandboxPendingChoice(null);
            // Bump sandbox state version so the AI turn loop
            // and any derived view models observe the post-
            // decision state (including advancing to an AI
            // turn after line/territory decisions).
            setSandboxStateVersion((v) => v + 1);
          }}
        />

        <main className="flex flex-col md:flex-row md:space-x-8 space-y-4 md:space-y-0">
          <section className="flex justify-center md:block">
            {sandboxBoardState && (
              <div className="inline-block space-y-3">
                <div className="p-4 rounded-2xl border border-slate-700 bg-slate-900/70 shadow-lg">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <p className="text-xs uppercase tracking-wide text-slate-400">
                        Local Sandbox
                      </p>
                      <h1 className="text-2xl font-bold text-white">Game – {boardDisplayLabel}</h1>
                    </div>
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={() => setShowSaveStateDialog(true)}
                        className="px-3 py-1 rounded-lg border border-slate-600 text-xs font-semibold text-slate-100 hover:border-sky-400 hover:text-sky-200 transition"
                      >
                        Save State
                      </button>
                      <button
                        type="button"
                        onClick={handleCopySandboxFixture}
                        className="px-3 py-1 rounded-lg border border-slate-600 text-xs font-semibold text-slate-100 hover:border-emerald-400 hover:text-emerald-200 transition"
                      >
                        Copy Test Fixture
                      </button>
                      <button
                        type="button"
                        onClick={() => setShowScenarioPicker(true)}
                        className="px-3 py-1 rounded-lg border border-slate-600 text-xs font-semibold text-slate-100 hover:border-amber-400 hover:text-amber-200 transition"
                      >
                        Load Scenario
                      </button>
                      {lastLoadedScenario && (
                        <button
                          type="button"
                          onClick={handleResetScenario}
                          className="px-3 py-1 rounded-lg border border-slate-600 text-xs font-semibold text-slate-100 hover:border-emerald-400 hover:text-emerald-200 transition"
                        >
                          Reset Scenario
                        </button>
                      )}
                      <button
                        type="button"
                        onClick={() => {
                          resetSandboxEngine();
                          setSelected(undefined);
                          setValidTargets([]);
                          setBackendSandboxError(null);
                          setSandboxPendingChoice(null);
                          setSandboxStallWarning(null);
                          setSandboxLastProgressAt(null);
                          setIsSandboxVictoryModalDismissed(false);
                        }}
                        className="px-3 py-1 rounded-lg border border-slate-600 text-xs font-semibold text-slate-100 hover:border-emerald-400 hover:text-emerald-200 transition"
                      >
                        Change Setup
                      </button>
                      <button
                        type="button"
                        aria-label="Show board controls"
                        data-testid="board-controls-button"
                        onClick={() => setShowBoardControls(true)}
                        className="h-8 w-8 rounded-full border border-slate-600 text-[11px] leading-none text-slate-200 hover:bg-slate-800/80"
                      >
                        ?
                      </button>
                    </div>
                  </div>
                </div>

                {isInReplayMode && (
                  <div className="mb-2 p-2 rounded-lg bg-emerald-900/40 border border-emerald-700/50 text-xs text-emerald-200 flex items-center justify-between">
                    <span>Viewing replay - board is read-only</span>
                    <span className="text-emerald-400/70">Use playback controls in sidebar</span>
                  </div>
                )}

                {sandboxBoardState && sandboxBoardViewModel && (
                  <BoardView
                    boardType={sandboxBoardState.type}
                    board={sandboxBoardState}
                    viewModel={sandboxBoardViewModel}
                    selectedPosition={isInReplayMode ? undefined : selected}
                    validTargets={isInReplayMode ? [] : displayedValidTargets}
                    onCellClick={isInReplayMode ? undefined : (pos) => handleSandboxCellClick(pos)}
                    onCellDoubleClick={
                      isInReplayMode ? undefined : (pos) => handleSandboxCellDoubleClick(pos)
                    }
                    onCellContextMenu={
                      isInReplayMode ? undefined : (pos) => handleSandboxCellContextMenu(pos)
                    }
                    showMovementGrid={showMovementGrid}
                    showCoordinateLabels={
                      sandboxBoardState.type === 'square8' || sandboxBoardState.type === 'square19'
                    }
                    showLineOverlays={true}
                    showTerritoryRegionOverlays={true}
                    pendingAnimation={
                      isInReplayMode
                        ? (replayAnimation ?? undefined)
                        : (pendingAnimation ?? undefined)
                    }
                    onAnimationComplete={
                      isInReplayMode ? () => setReplayAnimation(null) : clearAnimation
                    }
                  />
                )}

                <section className="mt-1 p-3 rounded-2xl border border-slate-700 bg-slate-900/70 shadow-lg flex flex-col lg:flex-row lg:items-center lg:justify-between gap-3 text-xs text-slate-200">
                  <div className="flex flex-wrap items-center gap-2">
                    {(() => {
                      let primarySubtitleText = boardDisplaySubtitle;
                      let primarySubtitleClass =
                        'px-2 py-1 rounded-full bg-slate-800/80 border border-slate-600';

                      if (isRingEliminationChoice) {
                        primarySubtitleText = 'Select stack cap to eliminate';
                        primarySubtitleClass =
                          'px-2 py-1 rounded-full bg-amber-500 text-slate-950 font-semibold border border-amber-300 shadow-sm shadow-amber-500/40';
                      } else if (isChainCaptureContinuationStep) {
                        primarySubtitleText = 'Continue Chain Capture';
                        primarySubtitleClass =
                          'px-2 py-1 rounded-full bg-amber-500 text-slate-950 font-semibold border border-amber-300 shadow-sm shadow-amber-500/40';
                      }

                      return <span className={primarySubtitleClass}>{primarySubtitleText}</span>;
                    })()}
                    <span className="px-2 py-1 rounded-full bg-slate-800/80 border border-slate-600">
                      Players: {config.numPlayers} ({humanSeatCount} human, {aiSeatCount} AI)
                    </span>
                    <span className="px-2 py-1 rounded-full bg-slate-800/80 border border-slate-600 min-w-[10rem] inline-flex justify-center text-center">
                      Phase: {sandboxPhaseDetails.label}
                    </span>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {sandboxPlayersList.map((player) => {
                      const typeKey = player.type === 'ai' ? 'ai' : 'human';
                      const meta = PLAYER_TYPE_META[typeKey as LocalPlayerType];
                      const isCurrent = player.playerNumber === sandboxCurrentPlayerNumber;
                      const nameLabel = player.username || `Player ${player.playerNumber}`;
                      return (
                        <span
                          key={player.playerNumber}
                          className={`px-3 py-1 rounded-full border transition ${
                            isCurrent ? 'border-white text-white bg-white/15' : meta.chip
                          }`}
                        >
                          P{player.playerNumber} • {nameLabel} ({meta.label})
                        </span>
                      );
                    })}
                  </div>
                </section>
              </div>
            )}
          </section>

          <aside className="w-full md:w-80 space-y-4 text-sm text-slate-100">
            {/* Replay Panel - Game Database Browser */}
            <ReplayPanel
              onStateChange={setReplayState}
              onReplayModeChange={setIsInReplayMode}
              onForkFromPosition={handleForkFromReplay}
              onAnimationChange={setReplayAnimation}
              defaultCollapsed={true}
            />

            <div className="p-4 border border-slate-700 rounded-2xl bg-slate-900/60 space-y-3">
              <h2 className="font-semibold">Players</h2>
              <div className="space-y-2">
                {sandboxHudViewModel.players.map((player) => (
                  <div
                    key={player.playerNumber}
                    className={`rounded-xl border px-3 py-2 text-xs flex items-center justify-between ${
                      player.isCurrent
                        ? 'border-emerald-400 bg-emerald-900/20'
                        : 'border-slate-700 bg-slate-900/40'
                    }`}
                  >
                    <div>
                      <p className="font-semibold text-white">
                        P{player.playerNumber} {player.username ? `• ${player.username}` : ''}
                      </p>
                      <p className="text-[11px] text-slate-400">
                        {player.type === 'ai' ? 'Computer' : 'Human'}
                      </p>
                    </div>
                    <div className="flex gap-3 text-right">
                      <div>
                        <p className="text-sm font-bold text-white">{player.ringsInHand}</p>
                        <p className="text-[11px] text-slate-400">in hand</p>
                      </div>
                      <div>
                        <p className="text-sm font-bold text-white">{player.territorySpaces}</p>
                        <p className="text-[11px] text-slate-400">territory</p>
                      </div>
                      <div>
                        <p className="text-sm font-bold text-white">{player.eliminatedRings}</p>
                        <p className="text-[11px] text-slate-400">eliminated</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {sandboxEngine &&
              sandboxGameState &&
              sandboxGameState.gameStatus === 'active' &&
              sandboxGameState.players.length === 2 &&
              sandboxGameState.rulesOptions?.swapRuleEnabled === true &&
              sandboxEngine.canCurrentPlayerSwapSides() && (
                <div className="p-3 border border-amber-500/60 rounded-2xl bg-amber-900/40 text-xs space-y-2">
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-semibold text-amber-100">
                      Pie rule available: swap colours with Player 1.
                    </span>
                    <button
                      type="button"
                      className="px-2 py-1 rounded bg-amber-500 hover:bg-amber-400 text-black font-semibold"
                      onClick={() => {
                        sandboxEngine.applySwapSidesForCurrentPlayer();
                        setSelected(undefined);
                        setValidTargets([]);
                        setSandboxPendingChoice(null);
                        setSandboxStateVersion((v) => v + 1);
                      }}
                    >
                      Swap colours
                    </button>
                  </div>
                  <p className="text-amber-100/80">
                    As Player 2, you may use this once, immediately after Player 1’s first turn.
                  </p>
                </div>
              )}

            {/* Move History - compact notation display */}
            {sandboxGameState && (
              <MoveHistory
                moves={sandboxGameState.moveHistory}
                boardType={sandboxGameState.boardType}
                currentMoveIndex={sandboxGameState.moveHistory.length - 1}
              />
            )}

            <div className="p-4 border border-slate-700 rounded-2xl bg-slate-900/60">
              <GameEventLog viewModel={sandboxEventLogViewModel} />
            </div>

            {/* Recording Status Panel */}
            <div className="p-3 border border-slate-700 rounded-2xl bg-slate-900/60">
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <span className="text-xs text-slate-400">Recording:</span>
                  {gameSaveStatus === 'idle' && autoSaveGames && (
                    <span className="flex items-center gap-1 text-xs text-slate-400">
                      <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                      Ready
                    </span>
                  )}
                  {gameSaveStatus === 'idle' && !autoSaveGames && (
                    <span className="text-xs text-slate-500">Disabled</span>
                  )}
                  {gameSaveStatus === 'saving' && (
                    <span className="flex items-center gap-1 text-xs text-amber-400">
                      <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
                      Saving...
                    </span>
                  )}
                  {gameSaveStatus === 'saved' && (
                    <span className="flex items-center gap-1 text-xs text-emerald-400">
                      <span className="w-2 h-2 rounded-full bg-emerald-400" />
                      Saved to server
                    </span>
                  )}
                  {gameSaveStatus === 'saved-local' && (
                    <span className="flex items-center gap-1 text-xs text-amber-300">
                      <span className="w-2 h-2 rounded-full bg-amber-300" />
                      Saved locally
                    </span>
                  )}
                  {gameSaveStatus === 'error' && (
                    <span className="flex items-center gap-1 text-xs text-red-400">
                      <span className="w-2 h-2 rounded-full bg-red-400" />
                      Failed
                    </span>
                  )}
                </div>
                <button
                  type="button"
                  onClick={() => setAutoSaveGames(!autoSaveGames)}
                  className={`px-2 py-0.5 rounded text-[10px] font-medium transition ${
                    autoSaveGames
                      ? 'bg-emerald-900/40 text-emerald-300 border border-emerald-700'
                      : 'bg-slate-800 text-slate-400 border border-slate-600'
                  }`}
                  title={autoSaveGames ? 'Click to disable recording' : 'Click to enable recording'}
                >
                  {autoSaveGames ? 'ON' : 'OFF'}
                </button>
              </div>
              {pendingLocalGames > 0 && (
                <div className="mt-2 flex items-center justify-between gap-2">
                  <div className="flex items-center gap-1.5">
                    {syncState?.status === 'syncing' ? (
                      <span className="w-2 h-2 rounded-full bg-blue-400 animate-pulse" />
                    ) : syncState?.status === 'offline' ? (
                      <span className="w-2 h-2 rounded-full bg-slate-500" />
                    ) : syncState?.status === 'error' ? (
                      <span className="w-2 h-2 rounded-full bg-red-400" />
                    ) : (
                      <span className="w-2 h-2 rounded-full bg-amber-400" />
                    )}
                    <span className="text-[10px] text-amber-400">
                      {pendingLocalGames} game{pendingLocalGames !== 1 ? 's' : ''}{' '}
                      {syncState?.status === 'syncing'
                        ? 'syncing...'
                        : syncState?.status === 'offline'
                          ? '(offline)'
                          : 'pending'}
                    </span>
                  </div>
                  <button
                    type="button"
                    onClick={() => GameSyncService.triggerSync()}
                    disabled={syncState?.status === 'syncing' || syncState?.status === 'offline'}
                    className="px-2 py-0.5 rounded text-[10px] font-medium bg-blue-900/40 text-blue-300 border border-blue-700 hover:bg-blue-800/40 disabled:opacity-50 disabled:cursor-not-allowed transition"
                    title="Sync pending games to server"
                  >
                    Sync
                  </button>
                </div>
              )}
            </div>

            <div className="p-4 border border-slate-700 rounded-2xl bg-slate-900/60 space-y-3">
              <div className="flex items-center justify-between gap-2">
                <h2 className="font-semibold">AI Evaluation (sandbox)</h2>
                <button
                  type="button"
                  onClick={requestSandboxEvaluation}
                  disabled={!sandboxEngine || !sandboxGameState || isSandboxAnalysisRunning}
                  className="px-3 py-1 rounded-full border border-slate-600 text-xs font-semibold text-slate-100 hover:border-emerald-400 hover:text-emerald-200 disabled:opacity-60 disabled:cursor-not-allowed transition"
                >
                  {isSandboxAnalysisRunning ? 'Evaluating…' : 'Request evaluation'}
                </button>
              </div>
              <EvaluationPanel
                evaluationHistory={sandboxEvaluationHistory}
                players={sandboxGameState?.players ?? []}
              />
            </div>

            <SandboxTouchControlsPanel
              selectedPosition={selected}
              selectedStackDetails={selectedStackDetails}
              validTargets={primaryValidTargets}
              isCaptureDirectionPending={!!sandboxCaptureChoice}
              captureTargets={sandboxCaptureTargets}
              // Multi-segment capture undo is not yet exposed by the sandbox
              // engine; this remains a no-op until the underlying rules
              // pipeline supports segment-level rewind.
              canUndoSegment={false}
              onClearSelection={() => {
                clearSandboxSelection();
              }}
              onUndoSegment={() => {
                // no-op for now
              }}
              // For now, treat "Finish move" as an explicit selection reset that
              // clears highlights without issuing additional engine actions.
              onApplyMove={() => {
                clearSandboxSelection();
              }}
              showMovementGrid={showMovementGrid}
              onToggleMovementGrid={(next) => setShowMovementGrid(next)}
              showValidTargets={showValidTargetsOverlay}
              onToggleValidTargets={(next) => setShowValidTargetsOverlay(next)}
              phaseLabel={sandboxPhaseDetails.label}
              autoSaveGames={autoSaveGames}
              onToggleAutoSave={(next) => setAutoSaveGames(next)}
              gameSaveStatus={gameSaveStatus}
            />

            <div className="p-4 border border-slate-700 rounded-2xl bg-slate-900/60 space-y-2">
              <h2 className="font-semibold">Phase Guide</h2>
              <p className="text-xs uppercase tracking-wide text-slate-400">
                {sandboxHudViewModel.phaseDetails.label}
              </p>
              <p className="text-sm text-slate-200">{sandboxHudViewModel.phaseDetails.summary}</p>
              <p className="text-xs text-slate-400">
                Complete the current requirement to advance the turn (chain captures, line rewards,
                etc.).
              </p>
            </div>

            <div className="p-4 border border-slate-700 rounded-2xl bg-slate-900/60 space-y-2">
              <h2 className="font-semibold">Sandbox Notes</h2>
              <ul className="list-disc list-inside text-slate-300 space-y-1 text-xs">
                {sandboxHudViewModel.modeNotes.map((note, idx) => (
                  <li key={idx}>{note}</li>
                ))}
              </ul>
            </div>
          </aside>
        </main>

        {showBoardControls && (
          <BoardControlsOverlay
            mode="sandbox"
            hasTouchControlsPanel
            onClose={() => setShowBoardControls(false)}
          />
        )}

        <ScenarioPickerModal
          isOpen={showScenarioPicker}
          onClose={() => setShowScenarioPicker(false)}
          onSelectScenario={handleLoadScenario}
        />

        <SaveStateDialog
          isOpen={showSaveStateDialog}
          onClose={() => setShowSaveStateDialog(false)}
          gameState={sandboxGameState}
          onSaved={(scenario) => {
            toast.success(`Saved state: ${scenario.name}`);
          }}
        />
      </div>
    </div>
  );
};

/**
 * Progress reporter utility for long-running TypeScript processes.
 *
 * Provides time-based progress output at configurable intervals (default: ~10s).
 * Extended with vector family tracking for orchestrator soak runs.
 */

/**
 * Format a duration in milliseconds as a human-readable string.
 */
function formatDuration(ms: number): string {
  const seconds = Math.floor(ms / 1000);
  if (seconds < 0) return '0s';
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) {
    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${minutes}m${secs}s`;
  }
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  return `${hours}h${minutes}m`;
}

export interface SoakProgressMetrics {
  gamesCompleted: number;
  totalMoves: number;
  totalDecisions?: number;
  gameDurations: number[];
  gameLengths: number[];
}

export interface VectorFamilyMetrics {
  gamesCompleted: number;
  totalMoves: number;
  gameDurations: number[];
  gameLengths: number[];
}

export class SoakProgressReporter {
  private startTime: number;
  private lastReportTime: number;
  private reportIntervalMs: number;
  private contextLabel: string;
  private currentActivity: string;
  private totalGames: number;
  private metrics: SoakProgressMetrics;
  private progressTimer: ReturnType<typeof setInterval> | null;
  private finished: boolean;

  constructor(options: { totalGames: number; reportIntervalSec?: number; contextLabel?: string }) {
    this.totalGames = options.totalGames;
    this.reportIntervalMs = (options.reportIntervalSec ?? 10) * 1000;
    this.contextLabel = options.contextLabel ?? '';
    this.currentActivity = '';
    this.startTime = Date.now();
    this.lastReportTime = this.startTime;
    this.metrics = {
      gamesCompleted: 0,
      totalMoves: 0,
      totalDecisions: 0,
      gameDurations: [],
      gameLengths: [],
    };
    this.finished = false;

    // Start background timer for guaranteed progress output every interval
    this.progressTimer = setInterval(() => {
      if (!this.finished) {
        this.emitReport();
        this.lastReportTime = Date.now();
      }
    }, this.reportIntervalMs);

    // Ensure timer doesn't prevent Node.js from exiting
    if (this.progressTimer.unref) {
      this.progressTimer.unref();
    }
  }

  /**
   * Set the current activity description for the next report.
   */
  setActivity(activity: string): void {
    this.currentActivity = activity;
  }

  /**
   * Check if a report is due and emit it if so.
   */
  check(): void {
    const now = Date.now();
    if (now - this.lastReportTime >= this.reportIntervalMs) {
      this.emitReport();
      this.lastReportTime = now;
    }
  }

  /**
   * Record completion of a game and potentially emit a progress report.
   */
  recordGame(gameData: {
    moves: number;
    durationMs: number;
    decisions?: number;
    forceReport?: boolean;
  }): void {
    this.metrics.gamesCompleted += 1;
    this.metrics.totalMoves += gameData.moves;
    this.metrics.totalDecisions = (this.metrics.totalDecisions ?? 0) + (gameData.decisions ?? 0);
    this.metrics.gameDurations.push(gameData.durationMs);
    this.metrics.gameLengths.push(gameData.moves);

    this.check();
  }

  private emitReport(): void {
    const now = Date.now();
    const elapsedMs = now - this.startTime;
    const elapsedSec = elapsedMs / 1000;
    const games = this.metrics.gamesCompleted;
    const total = this.totalGames;
    const remaining = Math.max(0, total - games);

    // Calculate rates
    const gamesPerSec = games / (elapsedSec || 1);
    const movesPerSec = this.metrics.totalMoves / (elapsedSec || 1);

    // Calculate averages
    const avgMovesPerGame = games > 0 ? this.metrics.totalMoves / games : 0;
    const avgMsPerGame =
      games > 0 ? this.metrics.gameDurations.reduce((a, b) => a + b, 0) / games : 0;

    // ETA
    const etaMs = gamesPerSec > 0 ? (remaining / gamesPerSec) * 1000 : 0;

    // Percentage
    const pct = total > 0 ? (games / total) * 100 : 0;

    const parts: string[] = [];

    if (this.contextLabel) {
      parts.push(`[${this.contextLabel}]`);
    }

    parts.push('PROGRESS:');
    if (this.currentActivity) {
      parts.push(`(${this.currentActivity})`);
    }
    parts.push(`${games}/${total} games (${pct.toFixed(1)}%)`);
    parts.push(`| elapsed: ${formatDuration(elapsedMs)}`);
    parts.push(`| ETA: ${formatDuration(etaMs)}`);
    parts.push(`| games/sec: ${gamesPerSec.toFixed(2)}`);
    parts.push(`| moves/sec: ${movesPerSec.toFixed(1)}`);
    parts.push(`| avg moves/game: ${avgMovesPerGame.toFixed(1)}`);
    parts.push(`| avg sec/game: ${(avgMsPerGame / 1000).toFixed(2)}`);

    // eslint-disable-next-line no-console
    console.log(parts.join(' '));
  }

  /**
   * Stop the background progress timer.
   */
  private stopTimer(): void {
    if (this.progressTimer) {
      clearInterval(this.progressTimer);
      this.progressTimer = null;
    }
  }

  /**
   * Emit a final summary report.
   */
  finish(): void {
    this.finished = true;
    this.stopTimer();
    const now = Date.now();
    const elapsedMs = now - this.startTime;
    const elapsedSec = elapsedMs / 1000;
    const games = this.metrics.gamesCompleted;

    const gamesPerSec = games / (elapsedSec || 1);
    const movesPerSec = this.metrics.totalMoves / (elapsedSec || 1);
    const avgMovesPerGame = games > 0 ? this.metrics.totalMoves / games : 0;
    const avgMsPerGame =
      games > 0 ? this.metrics.gameDurations.reduce((a, b) => a + b, 0) / games : 0;

    const parts: string[] = [];

    if (this.contextLabel) {
      parts.push(`[${this.contextLabel}]`);
    }

    parts.push('COMPLETED:');
    parts.push(`${games} games`);
    parts.push(`| total time: ${formatDuration(elapsedMs)}`);
    parts.push(`| total moves: ${this.metrics.totalMoves}`);
    parts.push(`| avg games/sec: ${gamesPerSec.toFixed(2)}`);
    parts.push(`| avg moves/sec: ${movesPerSec.toFixed(1)}`);
    parts.push(`| avg moves/game: ${avgMovesPerGame.toFixed(1)}`);
    parts.push(`| avg sec/game: ${(avgMsPerGame / 1000).toFixed(2)}`);

    // eslint-disable-next-line no-console
    console.log(parts.join(' '));
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Vector-Aware Soak Progress Reporter
// ═══════════════════════════════════════════════════════════════════════════

export class VectorAwareSoakProgressReporter {
  private startTime: number;
  private lastReportTime: number;
  private reportIntervalMs: number;
  private contextLabel: string;
  private currentActivity: string;
  private currentFamily: string;
  private totalGames: number;
  private metrics: SoakProgressMetrics;
  private familyMetrics: Map<string, VectorFamilyMetrics>;
  private progressTimer: ReturnType<typeof setInterval> | null;
  private finished: boolean;

  constructor(options: { totalGames: number; reportIntervalSec?: number; contextLabel?: string }) {
    this.totalGames = options.totalGames;
    this.reportIntervalMs = (options.reportIntervalSec ?? 10) * 1000;
    this.contextLabel = options.contextLabel ?? '';
    this.currentActivity = '';
    this.currentFamily = '';
    this.startTime = Date.now();
    this.lastReportTime = this.startTime;
    this.metrics = {
      gamesCompleted: 0,
      totalMoves: 0,
      totalDecisions: 0,
      gameDurations: [],
      gameLengths: [],
    };
    this.familyMetrics = new Map();
    this.finished = false;

    // Start background timer for guaranteed progress output every interval
    this.progressTimer = setInterval(() => {
      if (!this.finished) {
        this.emitReport();
        this.lastReportTime = Date.now();
      }
    }, this.reportIntervalMs);

    // Ensure timer doesn't prevent Node.js from exiting
    if (this.progressTimer.unref) {
      this.progressTimer.unref();
    }
  }

  /**
   * Set the current activity description for the next report.
   */
  setActivity(activity: string): void {
    this.currentActivity = activity;
  }

  /**
   * Set the current vector family being processed.
   */
  setCurrentFamily(family: string): void {
    this.currentFamily = family;
    if (!this.familyMetrics.has(family)) {
      this.familyMetrics.set(family, {
        gamesCompleted: 0,
        totalMoves: 0,
        gameDurations: [],
        gameLengths: [],
      });
    }
  }

  /**
   * Check if a report is due and emit it if so.
   */
  check(): void {
    const now = Date.now();
    if (now - this.lastReportTime >= this.reportIntervalMs) {
      this.emitReport();
      this.lastReportTime = now;
    }
  }

  /**
   * Record completion of a game and potentially emit a progress report.
   */
  recordGame(gameData: {
    moves: number;
    durationMs: number;
    decisions?: number;
    vectorFamily?: string;
    forceReport?: boolean;
  }): void {
    this.metrics.gamesCompleted += 1;
    this.metrics.totalMoves += gameData.moves;
    this.metrics.totalDecisions = (this.metrics.totalDecisions ?? 0) + (gameData.decisions ?? 0);
    this.metrics.gameDurations.push(gameData.durationMs);
    this.metrics.gameLengths.push(gameData.moves);

    // Update family metrics if applicable
    const family = gameData.vectorFamily ?? this.currentFamily;
    if (family) {
      let fm = this.familyMetrics.get(family);
      if (!fm) {
        fm = {
          gamesCompleted: 0,
          totalMoves: 0,
          gameDurations: [],
          gameLengths: [],
        };
        this.familyMetrics.set(family, fm);
      }
      fm.gamesCompleted += 1;
      fm.totalMoves += gameData.moves;
      fm.gameDurations.push(gameData.durationMs);
      fm.gameLengths.push(gameData.moves);
    }

    this.check();
  }

  private emitReport(): void {
    const now = Date.now();
    const elapsedMs = now - this.startTime;
    const elapsedSec = elapsedMs / 1000;
    const games = this.metrics.gamesCompleted;
    const total = this.totalGames;
    const remaining = Math.max(0, total - games);

    // Calculate rates
    const gamesPerSec = games / (elapsedSec || 1);
    const movesPerSec = this.metrics.totalMoves / (elapsedSec || 1);

    // Calculate averages
    const avgMovesPerGame = games > 0 ? this.metrics.totalMoves / games : 0;
    const avgMsPerGame =
      games > 0 ? this.metrics.gameDurations.reduce((a, b) => a + b, 0) / games : 0;

    // ETA
    const etaMs = gamesPerSec > 0 ? (remaining / gamesPerSec) * 1000 : 0;

    // Percentage
    const pct = total > 0 ? (games / total) * 100 : 0;

    const parts: string[] = [];

    if (this.contextLabel) {
      parts.push(`[${this.contextLabel}]`);
    }

    parts.push('PROGRESS:');

    // Include current family in progress output
    if (this.currentFamily) {
      parts.push(`[${this.currentFamily}]`);
    }

    if (this.currentActivity) {
      parts.push(`(${this.currentActivity})`);
    }
    parts.push(`${games}/${total} games (${pct.toFixed(1)}%)`);
    parts.push(`| elapsed: ${formatDuration(elapsedMs)}`);
    parts.push(`| ETA: ${formatDuration(etaMs)}`);
    parts.push(`| games/sec: ${gamesPerSec.toFixed(2)}`);
    parts.push(`| moves/sec: ${movesPerSec.toFixed(1)}`);
    parts.push(`| avg moves/game: ${avgMovesPerGame.toFixed(1)}`);
    parts.push(`| avg sec/game: ${(avgMsPerGame / 1000).toFixed(2)}`);

    // eslint-disable-next-line no-console
    console.log(parts.join(' '));
  }

  /**
   * Stop the background progress timer.
   */
  private stopTimer(): void {
    if (this.progressTimer) {
      clearInterval(this.progressTimer);
      this.progressTimer = null;
    }
  }

  /**
   * Emit a final summary report.
   */
  finish(): void {
    this.finished = true;
    this.stopTimer();
    const now = Date.now();
    const elapsedMs = now - this.startTime;
    const elapsedSec = elapsedMs / 1000;
    const games = this.metrics.gamesCompleted;

    const gamesPerSec = games / (elapsedSec || 1);
    const movesPerSec = this.metrics.totalMoves / (elapsedSec || 1);
    const avgMovesPerGame = games > 0 ? this.metrics.totalMoves / games : 0;
    const avgMsPerGame =
      games > 0 ? this.metrics.gameDurations.reduce((a, b) => a + b, 0) / games : 0;

    const parts: string[] = [];

    if (this.contextLabel) {
      parts.push(`[${this.contextLabel}]`);
    }

    parts.push('COMPLETED:');
    parts.push(`${games} games`);
    parts.push(`| total time: ${formatDuration(elapsedMs)}`);
    parts.push(`| total moves: ${this.metrics.totalMoves}`);
    parts.push(`| avg games/sec: ${gamesPerSec.toFixed(2)}`);
    parts.push(`| avg moves/sec: ${movesPerSec.toFixed(1)}`);
    parts.push(`| avg moves/game: ${avgMovesPerGame.toFixed(1)}`);
    parts.push(`| avg sec/game: ${(avgMsPerGame / 1000).toFixed(2)}`);

    // eslint-disable-next-line no-console
    console.log(parts.join(' '));

    // Print per-family summary if we have family metrics
    if (this.familyMetrics.size > 0) {
      // eslint-disable-next-line no-console
      console.log('');
      // eslint-disable-next-line no-console
      console.log('Per-Family Statistics:');
      for (const [family, fm] of this.familyMetrics.entries()) {
        const fmAvgMoves = fm.gamesCompleted > 0 ? fm.totalMoves / fm.gamesCompleted : 0;
        const fmAvgMs =
          fm.gamesCompleted > 0
            ? fm.gameDurations.reduce((a, b) => a + b, 0) / fm.gamesCompleted
            : 0;
        // eslint-disable-next-line no-console
        console.log(
          `  ${family}: ${fm.gamesCompleted} games, ${fm.totalMoves} moves, ` +
            `avg ${fmAvgMoves.toFixed(1)} moves/game, avg ${(fmAvgMs / 1000).toFixed(2)}s/game`
        );
      }
    }
  }

  /**
   * Get family metrics for reporting.
   */
  getFamilyMetrics(): Map<string, VectorFamilyMetrics> {
    return new Map(this.familyMetrics);
  }
}

export class OptimizationProgressReporter {
  private startTime: number;
  private lastReportTime: number;
  private reportIntervalMs: number;
  private totalGenerations: number;
  private candidatesPerGeneration: number;
  private currentGeneration: number;
  private candidatesEvaluated: number;
  private totalGamesPlayed: number;
  private bestFitness: number | null;

  constructor(options: {
    totalGenerations: number;
    candidatesPerGeneration: number;
    reportIntervalSec?: number;
  }) {
    this.totalGenerations = options.totalGenerations;
    this.candidatesPerGeneration = options.candidatesPerGeneration;
    this.reportIntervalMs = (options.reportIntervalSec ?? 10) * 1000;
    this.startTime = Date.now();
    this.lastReportTime = this.startTime;
    this.currentGeneration = 0;
    this.candidatesEvaluated = 0;
    this.totalGamesPlayed = 0;
    this.bestFitness = null;
  }

  get totalCandidates(): number {
    return this.totalGenerations * this.candidatesPerGeneration;
  }

  startGeneration(generation: number): void {
    this.currentGeneration = generation;
    // eslint-disable-next-line no-console
    console.log(`\n=== Generation ${generation}/${this.totalGenerations} ===`);
  }

  recordCandidate(data: {
    candidateIdx: number;
    fitness?: number;
    gamesPlayed?: number;
    forceReport?: boolean;
  }): void {
    this.candidatesEvaluated += 1;
    this.totalGamesPlayed += data.gamesPlayed ?? 0;

    if (data.fitness !== undefined) {
      if (this.bestFitness === null || data.fitness > this.bestFitness) {
        this.bestFitness = data.fitness;
      }
    }

    const now = Date.now();
    const elapsedSinceReport = now - this.lastReportTime;

    if (data.forceReport || elapsedSinceReport >= this.reportIntervalMs) {
      this.emitReport(data.candidateIdx);
      this.lastReportTime = now;
    }
  }

  private emitReport(currentCandidateIdx: number): void {
    const now = Date.now();
    const elapsedMs = now - this.startTime;
    const elapsedSec = elapsedMs / 1000;

    const totalCandidates = this.totalCandidates;
    const candidatesRemaining = Math.max(0, totalCandidates - this.candidatesEvaluated);

    const candidatesPerSec = this.candidatesEvaluated / (elapsedSec || 1);
    const gamesPerSec = this.totalGamesPlayed / (elapsedSec || 1);

    const etaMs = candidatesPerSec > 0 ? (candidatesRemaining / candidatesPerSec) * 1000 : 0;

    const overallPct = totalCandidates > 0 ? (this.candidatesEvaluated / totalCandidates) * 100 : 0;
    const genPct =
      this.candidatesPerGeneration > 0
        ? (currentCandidateIdx / this.candidatesPerGeneration) * 100
        : 0;

    const parts: string[] = [];

    parts.push(`[Gen ${this.currentGeneration}]`);
    parts.push('PROGRESS:');
    parts.push(
      `candidate ${currentCandidateIdx}/${this.candidatesPerGeneration} (${genPct.toFixed(0)}%)`
    );
    parts.push(
      `| overall: ${this.candidatesEvaluated}/${totalCandidates} (${overallPct.toFixed(1)}%)`
    );
    parts.push(`| elapsed: ${formatDuration(elapsedMs)}`);
    parts.push(`| ETA: ${formatDuration(etaMs)}`);
    if (this.totalGamesPlayed > 0) {
      parts.push(`| games/sec: ${gamesPerSec.toFixed(2)}`);
    }
    if (this.bestFitness !== null) {
      parts.push(`| best fitness: ${this.bestFitness.toFixed(4)}`);
    }

    // eslint-disable-next-line no-console
    console.log(parts.join(' '));
  }

  finishGeneration(stats: { meanFitness: number; bestFitness: number; stdFitness: number }): void {
    // eslint-disable-next-line no-console
    console.log(
      `[Gen ${this.currentGeneration}] SUMMARY: ` +
        `mean=${stats.meanFitness.toFixed(4)}, std=${stats.stdFitness.toFixed(4)}, best=${stats.bestFitness.toFixed(4)}`
    );
  }

  finish(): void {
    const now = Date.now();
    const elapsedMs = now - this.startTime;
    const elapsedSec = elapsedMs / 1000;

    const candidatesPerSec = this.candidatesEvaluated / (elapsedSec || 1);
    const gamesPerSec = this.totalGamesPlayed / (elapsedSec || 1);

    const parts: string[] = [];

    parts.push('OPTIMIZATION COMPLETED:');
    parts.push(`${this.currentGeneration} generations`);
    parts.push(`| ${this.candidatesEvaluated} candidates evaluated`);
    parts.push(`| total time: ${formatDuration(elapsedMs)}`);
    if (this.totalGamesPlayed > 0) {
      parts.push(`| total games: ${this.totalGamesPlayed}`);
      parts.push(`| avg games/sec: ${gamesPerSec.toFixed(2)}`);
    }
    parts.push(`| avg candidates/sec: ${candidatesPerSec.toFixed(3)}`);
    if (this.bestFitness !== null) {
      parts.push(`| best fitness: ${this.bestFitness.toFixed(4)}`);
    }

    // eslint-disable-next-line no-console
    console.log(parts.join(' '));
  }
}

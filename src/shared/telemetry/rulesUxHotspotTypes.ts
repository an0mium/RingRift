import type { RulesUxContext, RulesUxSource } from './rulesUxEvents';

/**
 * Aggregate counts of rules‑UX telemetry events for a single surface ({@link RulesUxSource}).
 *
 * This is a purely offline analysis format derived from the Prometheus counter
 * `ringrift_rules_ux_events_total`. Each entry corresponds to one
 * (rulesContext, source) pair collapsed into a simple `eventType -> count`
 * map for a given analysis window.
 *
 * The keys of {@link events} are usually a subset of the {@link RulesUxEventType}
 * values defined in {@link rulesUxEvents.ts}, but the analyzer keeps this map
 * generic so that future pipelines can add derived counters without changing
 * the schema.
 */
export interface RulesUxSourceAggregate {
  /** Surface that emitted the events (HUD, victory modal, teaching overlay, etc.). */
  source: RulesUxSource | string;

  /**
   * Map from event type identifier to aggregated count for the analysis window.
   *
   * Expected keys for the current analyzer include (when present):
   *
   * - `help_open`
   * - `help_reopen`
   * - `rules_help_repeat`              (legacy alias for help reopen)
   * - `weird_state_banner_impression`
   * - `weird_state_details_open`
   * - `resign_after_weird_state`
   * - `rules_weird_state_resign`       (legacy alias for resign-after-weird)
   *
   * Additional keys are allowed and ignored by the hotspot analyzer.
   */
  events: { [eventType: string]: number };
}

/**
 * Aggregated rules‑UX counts for a single semantic rules context
 * (e.g. `"anm_forced_elimination"`).
 */
export interface RulesUxContextAggregate {
  /** Low-cardinality semantic rules context. */
  rulesContext: RulesUxContext | string;

  /** Per-source breakdown for this context. */
  sources: RulesUxSourceAggregate[];
}

/**
 * Root object for a pre-aggregated rules‑UX snapshot used by the hotspot analyzer.
 *
 * This JSON can be produced from Prometheus or a data warehouse and then
 * consumed by {@link analyze_rules_ux_telemetry.ts} under `scripts/`.
 *
 * NOTE: This file is intentionally decoupled from the live telemetry event
 * payloads; it models offline, derived aggregates only. Do not import it
 * into runtime request handlers.
 */
export interface RulesUxAggregatesRoot {
  /** Board topology for the aggregated data (currently "square8"). */
  board: string;

  /** Number of players in the aggregated games (currently 2). */
  num_players: number;

  /** Analysis window in UTC. */
  window: {
    start: string;
    end: string;
  };

  /** Aggregate game counters for the same window. */
  games: {
    started: number;
    completed: number;
  };

  /** Per-context aggregates. Must be a non-empty array for a valid snapshot. */
  contexts: RulesUxContextAggregate[];
}

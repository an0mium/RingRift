import { getScenarioById, TerritoryRuleScenario } from './rulesMatrix';

/**
 * RulesMatrix → mini-region territory scenario (square8, Q23 numeric invariant)
 *
 * This is a light-weight, data-only check that the compact Q23 mini-region
 * scenario is correctly encoded in the shared rules matrix and accessible via
 * getScenarioById. Numeric invariants for this geometry are asserted in the
 * dedicated rules-layer tests:
 *
 *   - tests/unit/territoryProcessing.rules.test.ts
 *   - tests/unit/sandboxTerritory.rules.test.ts
 *   - tests/unit/sandboxTerritoryEngine.rules.test.ts
 *
 * Scenario ID:
 *   Rules_12_2_Q23_mini_region_square8_numeric_invariant
 */

describe('RulesMatrix → TerritoryRuleScenario – Q23 mini-region numeric invariant (square8)', () => {
  it('exposes the compact Q23 mini-region scenario via getScenarioById', () => {
    const id = 'Rules_12_2_Q23_mini_region_square8_numeric_invariant';
    const scenario = getScenarioById(id) as TerritoryRuleScenario | undefined;

    expect(scenario).toBeDefined();
    if (!scenario) return;

    expect(scenario.kind).toBe('territory');
    expect(scenario.boardType).toBe('square8');
    expect(scenario.movingPlayer).toBe(1);
    expect(scenario.ref.id).toBe(id);
    expect(scenario.ref.rulesSections).toContain('§12.2');
    expect(scenario.ref.faqRefs).toContain('Q23');

    expect(scenario.regions.length).toBe(1);
    const [region] = scenario.regions;

    // Geometry: 2×2 mini-region at (2,2)–(3,3) containing victim stacks for player 2.
    const expectedSpaces = [
      { x: 2, y: 2 },
      { x: 2, y: 3 },
      { x: 3, y: 2 },
      { x: 3, y: 3 },
    ];

    expect(region.spaces).toEqual(expectedSpaces);
    expect(region.controllingPlayer).toBe(1);
    expect(region.victimPlayer).toBe(2);
    expect(region.movingPlayerHasOutsideStack).toBe(true);
    expect(region.outsideStackPosition).toEqual({ x: 0, y: 0 });
    expect(region.selfEliminationStackHeight).toBe(3);
  });

  it('exposes the Q20 region-order two-region scenario via getScenarioById', () => {
    const id = 'Rules_12_3_region_order_choice_two_regions_square8';
    const scenario = getScenarioById(id) as TerritoryRuleScenario | undefined;

    expect(scenario).toBeDefined();
    if (!scenario) return;

    expect(scenario.kind).toBe('territory');
    expect(scenario.boardType).toBe('square8');
    expect(scenario.movingPlayer).toBe(1);
    expect(scenario.ref.id).toBe(id);
    expect(scenario.ref.rulesSections).toContain('§12.3');
    expect(scenario.ref.faqRefs).toContain('Q20');

    // Two disconnected regions should be surfaced to the moving player for a
    // RegionOrderChoice-style decision.
    expect(scenario.regions.length).toBe(2);

    const [regionA, regionB] = scenario.regions;

    const expectedRegionASpaces = [
      { x: 1, y: 1 },
      { x: 1, y: 2 },
    ];
    const expectedRegionBSpaces = [
      { x: 5, y: 5 },
      { x: 5, y: 6 },
    ];

    expect(regionA.spaces).toEqual(expectedRegionASpaces);
    expect(regionB.spaces).toEqual(expectedRegionBSpaces);

    // These regions are neutral in terms of controlling/victim players; the
    // key semantics (region-order decision and self-elimination prerequisite)
    // are asserted by the dedicated integration suites:
    //   - tests/unit/GameEngine.regionOrderChoiceIntegration.test.ts
    //   - tests/unit/ClientSandboxEngine.regionOrderChoice.test.ts
    expect(regionA.controllingPlayer).toBe(0);
    expect(regionB.controllingPlayer).toBe(0);
    expect(regionA.victimPlayer).toBe(0);
    expect(regionB.victimPlayer).toBe(0);
    expect(regionA.movingPlayerHasOutsideStack).toBe(true);
    expect(regionB.movingPlayerHasOutsideStack).toBe(true);
  });
});

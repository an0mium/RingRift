import type {
  CuratedScenarioBundle,
  LoadableScenario,
} from '../../src/client/sandbox/scenarioTypes';
import { SCENARIO_RULES_CONCEPTS } from '../../src/client/sandbox/scenarioTypes';
import curatedBundle from '../../public/scenarios/curated.json';

const curatedScenarios = (curatedBundle as unknown as CuratedScenarioBundle)
  .scenarios as LoadableScenario[];

describe('Curated sandbox scenario metadata', () => {
  it('has required structural fields for every scenario', () => {
    expect(Array.isArray(curatedScenarios)).toBe(true);
    expect(curatedScenarios.length).toBeGreaterThan(0);

    for (const scenario of curatedScenarios) {
      expect(typeof scenario.id).toBe('string');
      expect(scenario.id).not.toHaveLength(0);

      expect(typeof scenario.name).toBe('string');
      expect(scenario.name).not.toHaveLength(0);

      expect(typeof scenario.description).toBe('string');
      expect(scenario.description).not.toHaveLength(0);

      expect(typeof scenario.category).toBe('string');

      expect(typeof scenario.boardType).toBe('string');
      expect(typeof scenario.playerCount).toBe('number');

      expect(typeof scenario.createdAt).toBe('string');
      expect(scenario.source).toBe('curated');
    }
  });

  it('uses only allowed rulesConcept values and non-empty uxSpecAnchor when present', () => {
    const allowedConcepts = new Set(SCENARIO_RULES_CONCEPTS);

    for (const scenario of curatedScenarios) {
      // All curated scenarios should declare a rulesConcept to tie them
      // back to UX_RULES_COPY_SPEC sections.
      expect(typeof scenario.rulesConcept).toBe('string');
      expect(scenario.rulesConcept).toBeTruthy();

      if (scenario.rulesConcept) {
        expect(allowedConcepts.has(scenario.rulesConcept)).toBe(true);
      }

      if (scenario.uxSpecAnchor !== undefined) {
        expect(typeof scenario.uxSpecAnchor).toBe('string');
        expect(scenario.uxSpecAnchor).not.toHaveLength(0);
      }
    }
  });

  it('avoids legacy or misleading wording in descriptions and rules snippets', () => {
    const forbiddenPatterns: RegExp[] = [
      /rings?\s+captured\s+to\s+hand/i,
      /captured\s+to\s+hand/i,
      /ring[s]?\s+to\s+hand/i,
      /exactly\s+equal\s+to\s+(its|the)\s+stack['’]s?\s+height/i,
      /move\s+exactly\s+\d+\s+spaces/i,
    ];

    for (const scenario of curatedScenarios) {
      const blob = [scenario.description ?? '', scenario.rulesSnippet ?? ''].join('\n');

      for (const pattern of forbiddenPatterns) {
        expect(pattern.test(blob)).toBe(false);
      }
    }
  });
});

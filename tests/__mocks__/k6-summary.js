/**
 * Jest mock for the remote k6 text summary helper used in tests/load/summary.js.
 *
 * In real k6 runs, `textSummary(data, options)` renders a human-readable
 * summary for stdout. For Jest unit tests we only need a deterministic,
 * side-effect-free implementation so that handleSummary() can attach a
 * string to `outputs.stdout` without performing any I/O or network work.
 *
 * This module is wired via the `moduleNameMapper` entry in jest.config.js:
 *
 *   '^https://jslib\\.k6\\.io/k6-summary/0\\.0\\.1/index\\.js$':
 *     '<rootDir>/tests/__mocks__/k6-summary.js',
 *
 * so that importing the remote URL inside tests/load/summary.js resolves
 * cleanly under Jest.
 */

/**
 * Minimal mock implementation of k6 `textSummary`.
 *
 * @param {any} data - k6 summary object (or a Jest fake)
 * @param {any} options - options passed by the caller (ignored)
 * @returns {string}
 */
function textSummary(data, options) {
  const metricCount =
    data && typeof data === 'object' && data.metrics
      ? Object.keys(data.metrics).length
      : 0;
  return `k6-summary-mock (metrics=${metricCount})`;
}

module.exports = {
  textSummary,
};
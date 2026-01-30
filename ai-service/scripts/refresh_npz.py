#!/usr/bin/env python3
"""Refresh stale NPZ files for specified configs."""
import subprocess
import sys
import os

os.environ['PYTHONPATH'] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['RINGRIFT_ALLOW_PENDING_GATE'] = '1'

CONFIGS = [
    ('hexagonal_2p', 'data/games/canonical_hexagonal_2p.db', 'hexagonal', '2'),
    ('hexagonal_3p', 'data/games/canonical_hexagonal_3p.db', 'hexagonal', '3'),
    ('square19_2p', 'data/games/canonical_square19_2p.db', 'square19', '2'),
]

def main():
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'export_replay_dataset.py')

    for name, db, board, np in CONFIGS:
        output = f'data/training/{name}_fresh.npz'
        print(f'\n=== Refreshing {name} ===')
        cmd = [
            sys.executable, script,
            '--db', db,
            '--board-type', board,
            '--num-players', np,
            '--output', output,
            '--force-export',
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        for line in result.stdout.strip().split('\n')[-10:]:
            print(line)
        if result.returncode != 0:
            for line in result.stderr.strip().split('\n')[-5:]:
                print(f'  STDERR: {line}')
        print(f'  Exit code: {result.returncode}')

if __name__ == '__main__':
    main()

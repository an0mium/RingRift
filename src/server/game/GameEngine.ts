import {
  GameState,
  Move,
  Player,
  BoardType,
  TimeControl,
  GameResult,
  BOARD_CONFIGS,
  Position,
  RingStack,
  Territory,
  LineInfo,
  positionToString
} from '../../shared/types/game';
import { BoardManager } from './BoardManager';
import { RuleEngine } from './RuleEngine';

// Timer functions for Node.js environment
declare const setTimeout: (callback: () => void, ms: number) => any;
declare const clearTimeout: (timer: any) => void;

// Using a simple UUID generator for now
function generateUUID(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    const r = Math.random() * 16 | 0;
    const v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}

export class GameEngine {
  private gameState: GameState;
  private boardManager: BoardManager;
  private ruleEngine: RuleEngine;
  private moveTimers: Map<number, any> = new Map();

  constructor(
    gameId: string,
    boardType: BoardType,
    players: Player[],
    timeControl: TimeControl,
    isRated: boolean = true
  ) {
    this.boardManager = new BoardManager(boardType);
    this.ruleEngine = new RuleEngine(this.boardManager, boardType);
    
    const config = BOARD_CONFIGS[boardType];
    
    this.gameState = {
      id: gameId,
      boardType,
      board: this.boardManager.createBoard(),
      players: players.map((p, index) => ({
        ...p,
        playerNumber: index + 1,
        timeRemaining: timeControl.initialTime * 1000, // Convert to milliseconds
        isReady: p.type === 'ai' // AI players are always ready
      })),
      currentPhase: 'ring_placement',
      currentPlayer: 1,
      moveHistory: [],
      timeControl,
      spectators: [],
      gameStatus: 'waiting',
      createdAt: new Date(),
      lastMoveAt: new Date(),
      isRated,
      maxPlayers: players.length,
      totalRingsInPlay: config.ringsPerPlayer * players.length,
      totalRingsEliminated: 0,
      victoryThreshold: Math.floor(config.ringsPerPlayer * players.length / 2) + 1,
      territoryVictoryThreshold: Math.floor(config.totalSpaces / 2) + 1
    };
  }

  getGameState(): GameState {
    return { ...this.gameState };
  }

  startGame(): boolean {
    // Check if all players are ready
    const allReady = this.gameState.players.every(p => p.isReady);
    if (!allReady) {
      return false;
    }

    this.gameState.gameStatus = 'active';
    this.gameState.lastMoveAt = new Date();
    
    // Start the first player's timer
    this.startPlayerTimer(this.gameState.currentPlayer);
    
    return true;
  }

  makeMove(move: Omit<Move, 'id' | 'timestamp' | 'moveNumber'>): {
    success: boolean;
    error?: string;
    gameState?: GameState;
    gameResult?: GameResult;
  } {
    // Validate the move
    const fullMove: Move = {
      ...move,
      id: generateUUID(),
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: this.gameState.moveHistory.length + 1
    };

    const validation = this.ruleEngine.validateMove(fullMove, this.gameState);
    if (!validation) {
      return {
        success: false,
        error: 'Invalid move'
      };
    }

    // Stop current player's timer
    this.stopPlayerTimer(this.gameState.currentPlayer);

    // Apply the move
    const moveResult = this.applyMove(fullMove);
    
    // Add move to history
    this.gameState.moveHistory.push(fullMove);
    this.gameState.lastMoveAt = new Date();

    // Process automatic consequences
    this.processAutomaticConsequences(moveResult);

    // Check for game end conditions
    const gameEndCheck = this.ruleEngine.checkGameEnd(this.gameState);
    if (gameEndCheck.isGameOver) {
      return this.endGame(gameEndCheck.winner, gameEndCheck.reason || 'unknown');
    }

    // Advance to next phase/player
    this.advanceGame();

    // Start next player's timer
    this.startPlayerTimer(this.gameState.currentPlayer);

    return {
      success: true,
      gameState: this.getGameState()
    };
  }

  private applyMove(move: Move): {
    captures: Position[];
    territoryChanges: Territory[];
    lineCollapses: LineInfo[];
  } {
    const result = {
      captures: [] as Position[],
      territoryChanges: [] as Territory[],
      lineCollapses: [] as LineInfo[]
    };

    switch (move.type) {
      case 'place_ring':
        if (move.to) {
          const newStack: RingStack = {
            position: move.to,
            stackHeight: 1,
            capHeight: 1,
            controllingPlayer: move.player,
            rings: [move.player]
          };
          this.boardManager.setStack(move.to, newStack, this.gameState.board);
          
          // Update player state: decrement rings in hand
          const player = this.gameState.players.find(p => p.playerNumber === move.player);
          if (player && player.ringsInHand > 0) {
            player.ringsInHand--;
          }
        }
        break;

      case 'move_ring':
        if (move.from && move.to) {
          const stack = this.boardManager.getStack(move.from, this.gameState.board);
          if (stack) {
            // Rule Reference: Section 4.2.1 - Leave marker on departure space
            this.boardManager.setMarker(move.from, move.player, this.gameState.board);
            
            // Process markers along movement path (Section 8.3)
            this.processMarkersAlongPath(move.from, move.to, move.player);
            
            // Check if landing on same-color marker (Section 8.2)
            const landingMarker = this.boardManager.getMarker(move.to, this.gameState.board);
            if (landingMarker === move.player) {
              this.boardManager.removeMarker(move.to, this.gameState.board);
            }
            
            // Remove stack from source
            this.boardManager.removeStack(move.from, this.gameState.board);
            
            // Normal movement (no capture at landing position)
            const movedStack: RingStack = {
              ...stack,
              position: move.to
            };
            this.boardManager.setStack(move.to, movedStack, this.gameState.board);
          }
        }
        break;

      case 'overtaking_capture':
        if (move.from && move.to && move.captureTarget) {
          const stack = this.boardManager.getStack(move.from, this.gameState.board);
          const targetStack = this.boardManager.getStack(move.captureTarget, this.gameState.board);
          
          if (stack && targetStack) {
            // Rule Reference: Section 10.2 - Overtaking Capture
            // Leave marker on departure space
            this.boardManager.setMarker(move.from, move.player, this.gameState.board);
            
            // Process markers along path to target
            this.processMarkersAlongPath(move.from, move.captureTarget, move.player);
            
            // Process markers along path from target to landing
            this.processMarkersAlongPath(move.captureTarget, move.to, move.player);
            
            // Check if landing on same-color marker
            const landingMarker = this.boardManager.getMarker(move.to, this.gameState.board);
            if (landingMarker === move.player) {
              this.boardManager.removeMarker(move.to, this.gameState.board);
            }
            
            // Capture top ring from target stack and add to bottom of capturing stack
            // Rule Reference: Section 10.2 - Top ring added to bottom
            const capturedRing = targetStack.rings[0]; // Top ring
            const newRings = [...stack.rings, capturedRing];
            
            // Update target stack (remove top ring)
            const remainingTargetRings = targetStack.rings.slice(1);
            if (remainingTargetRings.length > 0) {
              const newTargetStack: RingStack = {
                ...targetStack,
                rings: remainingTargetRings,
                stackHeight: remainingTargetRings.length,
                capHeight: this.calculateCapHeight(remainingTargetRings),
                controllingPlayer: remainingTargetRings[0]
              };
              this.boardManager.setStack(move.captureTarget, newTargetStack, this.gameState.board);
            } else {
              // Target stack is now empty, remove it
              this.boardManager.removeStack(move.captureTarget, this.gameState.board);
            }
            
            // Remove capturing stack from source
            this.boardManager.removeStack(move.from, this.gameState.board);
            
            // Place capturing stack at landing position with captured ring
            const newStack: RingStack = {
              position: move.to,
              rings: newRings,
              stackHeight: newRings.length,
              capHeight: this.calculateCapHeight(newRings),
              controllingPlayer: newRings[0]
            };
            this.boardManager.setStack(move.to, newStack, this.gameState.board);
            
            result.captures.push(move.captureTarget);
          }
        }
        break;

      case 'build_stack':
        if (move.from && move.to && move.buildAmount) {
          const sourceStack = this.boardManager.getStack(move.from, this.gameState.board);
          const targetStack = this.boardManager.getStack(move.to, this.gameState.board);
          
          if (sourceStack && targetStack && move.buildAmount) {
            // Transfer rings from source to target
            const transferRings = sourceStack.rings.slice(0, move.buildAmount);
            const remainingRings = sourceStack.rings.slice(move.buildAmount);
            
            const newSourceStack: RingStack = {
              ...sourceStack,
              stackHeight: sourceStack.stackHeight - move.buildAmount,
              rings: remainingRings
            };
            
            const newTargetStack: RingStack = {
              ...targetStack,
              stackHeight: targetStack.stackHeight + move.buildAmount,
              capHeight: Math.max(targetStack.capHeight, move.buildAmount),
              rings: [...targetStack.rings, ...transferRings]
            };
            
            // Update stacks
            if (newSourceStack.stackHeight > 0) {
              this.boardManager.setStack(move.from, newSourceStack, this.gameState.board);
            } else {
              this.boardManager.removeStack(move.from, this.gameState.board);
            }
            this.boardManager.setStack(move.to, newTargetStack, this.gameState.board);
          }
        }
        break;
    }

    // Line formation is processed separately in line_processing phase
    // This is just for tracking lines that were formed during the move

    // Check for territory disconnection
    const territories = this.boardManager.findAllTerritoriesForAllPlayers(this.gameState.board);
    for (const territory of territories) {
      if (territory.isDisconnected) {
        // Remove disconnected territory
        for (const pos of territory.spaces) {
          this.boardManager.removeStack(pos, this.gameState.board);
        }
        result.territoryChanges.push(territory);
      }
    }

    return result;
  }

  /**
   * Process automatic consequences after a move
   * Rule Reference: Section 4.5 - Post-Movement Processing
   */
  private processAutomaticConsequences(moveResult: {
    captures: Position[];
    territoryChanges: Territory[];
    lineCollapses: LineInfo[];
  }): void {
    // Captures are already processed in applyMove
    
    // Process line formations (Section 11.2, 11.3)
    this.processLineFormations();
    
    // Process territory disconnections (Section 12.2)
    this.processDisconnectedRegions();
  }

  /**
   * Process all line formations with graduated rewards
   * Rule Reference: Section 11.2, 11.3
   * 
   * For exact required length (4 for 8x8, 5 for 19x19/hex):
   *   - Collapse all markers
   *   - Eliminate one ring or cap from controlled stack
   * 
   * For longer lines (5+ for 8x8, 6+ for 19x19/hex):
   *   - Option 1: Collapse all + eliminate ring/cap
   *   - Option 2: Collapse required markers only, no elimination
   */
  private processLineFormations(): void {
    const config = BOARD_CONFIGS[this.gameState.boardType];
    
    // Keep processing until no more lines exist
    while (true) {
      const lines = this.boardManager.findAllLines(this.gameState.board);
      if (lines.length === 0) break;
      
      // Process each line for the moving player
      // TODO: In full implementation, moving player should choose which line to process first
      // For now, process in order found
      const line = lines[0];
      
      // Only process lines for the moving player
      if (line.player !== this.gameState.currentPlayer) {
        // Skip lines from other players (shouldn't happen in current rules)
        break;
      }
      
      this.processOneLine(line, config.lineLength);
      
      // After processing one line, re-check for remaining lines
      // (lines may have changed due to collapsed spaces)
    }
  }

  /**
   * Process a single line formation
   * Rule Reference: Section 11.2
   */
  private processOneLine(line: LineInfo, requiredLength: number): void {
    const lineLength = line.positions.length;
    
    if (lineLength === requiredLength) {
      // Exact required length: Must collapse all and eliminate ring/cap
      this.collapseLineMarkers(line.positions, line.player);
      this.eliminatePlayerRingOrCap(line.player);
    } else if (lineLength > requiredLength) {
      // Longer than required: Choose option (for now, always use Option 2 to preserve rings)
      // TODO: In full implementation, player should choose Option 1 or Option 2
      // Option 2: Collapse only required markers, no elimination
      const markersToCollapse = line.positions.slice(0, requiredLength);
      this.collapseLineMarkers(markersToCollapse, line.player);
      
      // Option 1 would be:
      // this.collapseLineMarkers(line.positions, line.player);
      // this.eliminatePlayerRingOrCap(line.player);
    }
  }

  /**
   * Collapse marker positions to player's color territory
   * Rule Reference: Section 11.2 - Markers collapse to colored spaces
   */
  private collapseLineMarkers(positions: Position[], player: number): void {
    for (const pos of positions) {
      this.boardManager.setCollapsedSpace(pos, player, this.gameState.board);
    }
    // Update player's territory count
    this.updatePlayerTerritorySpaces(player, positions.length);
  }

  /**
   * Eliminate one ring or cap from player's controlled stacks
   * Rule Reference: Section 11.2 - Moving player chooses which ring/stack cap to eliminate
   */
  private eliminatePlayerRingOrCap(player: number): void {
    const playerStacks = this.boardManager.getPlayerStacks(this.gameState.board, player);
    
    if (playerStacks.length === 0) {
      // No stacks to eliminate from, player might have rings in hand
      const playerState = this.gameState.players.find(p => p.playerNumber === player);
      if (playerState && playerState.ringsInHand > 0) {
        // Eliminate from hand
        playerState.ringsInHand--;
        this.gameState.totalRingsEliminated++;
        
        // Track eliminated rings in board state
        if (!this.gameState.board.eliminatedRings[player]) {
          this.gameState.board.eliminatedRings[player] = 0;
        }
        this.gameState.board.eliminatedRings[player]++;
        
        // Update player state
        this.updatePlayerEliminatedRings(player, 1);
      }
      return;
    }
    
    // TODO: In full implementation, player should choose which stack
    // For now, eliminate from first stack
    const stack = playerStacks[0];
    
    // Calculate cap height
    const capHeight = this.calculateCapHeight(stack.rings);
    
    // Eliminate the entire cap (all consecutive top rings of controlling color)
    const remainingRings = stack.rings.slice(capHeight);
    
    // Update eliminated rings count
    this.gameState.totalRingsEliminated += capHeight;
    if (!this.gameState.board.eliminatedRings[player]) {
      this.gameState.board.eliminatedRings[player] = 0;
    }
    this.gameState.board.eliminatedRings[player] += capHeight;
    
    // Update player state
    this.updatePlayerEliminatedRings(player, capHeight);
    
    if (remainingRings.length > 0) {
      // Update stack with remaining rings
      const newStack: RingStack = {
        ...stack,
        rings: remainingRings,
        stackHeight: remainingRings.length,
        capHeight: this.calculateCapHeight(remainingRings),
        controllingPlayer: remainingRings[0]
      };
      this.boardManager.setStack(stack.position, newStack, this.gameState.board);
    } else {
      // Stack is now empty, remove it
      this.boardManager.removeStack(stack.position, this.gameState.board);
    }
  }

  /**
   * Update player's eliminatedRings counter
   */
  private updatePlayerEliminatedRings(playerNumber: number, count: number): void {
    const player = this.gameState.players.find(p => p.playerNumber === playerNumber);
    if (player) {
      player.eliminatedRings += count;
    }
  }

  /**
   * Update player's territorySpaces counter
   */
  private updatePlayerTerritorySpaces(playerNumber: number, count: number): void {
    const player = this.gameState.players.find(p => p.playerNumber === playerNumber);
    if (player) {
      player.territorySpaces += count;
    }
  }

  /**
   * Process disconnected regions with chain reactions
   * Rule Reference: Section 12.2, 12.3 - Territory Disconnection and Chain Reactions
   */
  private processDisconnectedRegions(): void {
    const movingPlayer = this.gameState.currentPlayer;
    
    // Keep processing until no more disconnections occur
    while (true) {
      const disconnectedRegions = this.boardManager.findDisconnectedRegions(
        this.gameState.board,
        movingPlayer
      );
      
      if (disconnectedRegions.length === 0) break;
      
      // Process each region (player chooses order)
      // TODO: In full implementation, player should choose which region to process first
      // For now, process in order found
      const region = disconnectedRegions[0];
      
      // Self-elimination prerequisite check
      if (!this.canProcessDisconnectedRegion(region, movingPlayer)) {
        // Cannot process this region, skip it
        // In reality, if we can't process any regions, we should break
        // For now, just break to avoid infinite loop
        break;
      }
      
      // Process the disconnected region
      this.processOneDisconnectedRegion(region, movingPlayer);
    }
  }

  /**
   * Check if player can process a disconnected region
   * Rule Reference: Section 12.2 - Self-Elimination Prerequisite
   * 
   * Player must have at least one ring/cap outside the region before processing
   */
  private canProcessDisconnectedRegion(region: Territory, player: number): boolean {
    const regionPositionSet = new Set(region.spaces.map(pos => positionToString(pos)));
    const playerStacks = this.boardManager.getPlayerStacks(this.gameState.board, player);
    
    // Check if player has at least one ring/cap outside this region
    for (const stack of playerStacks) {
      const stackPosKey = positionToString(stack.position);
      if (!regionPositionSet.has(stackPosKey)) {
        // Found a stack outside the region
        return true;
      }
    }
    
    // No stacks outside the region - cannot process
    return false;
  }

  /**
   * Process a single disconnected region
   * Rule Reference: Section 12.2 - Processing steps
   */
  private processOneDisconnectedRegion(region: Territory, movingPlayer: number): void {
    // 1. Get border markers to collapse
    const borderMarkers = this.boardManager.getBorderMarkerPositions(
      region.spaces,
      this.gameState.board
    );
    
    // 2. Collapse all spaces in the region to moving player's color
    for (const pos of region.spaces) {
      this.boardManager.setCollapsedSpace(pos, movingPlayer, this.gameState.board);
    }
    
    // 3. Collapse all border markers to moving player's color
    for (const pos of borderMarkers) {
      this.boardManager.setCollapsedSpace(pos, movingPlayer, this.gameState.board);
    }
    
    // Update player's territory count (region spaces + border markers)
    const totalTerritoryGained = region.spaces.length + borderMarkers.length;
    this.updatePlayerTerritorySpaces(movingPlayer, totalTerritoryGained);
    
    // 4. Eliminate all rings within the region (all colors)
    let totalRingsEliminated = 0;
    for (const pos of region.spaces) {
      const stack = this.boardManager.getStack(pos, this.gameState.board);
      if (stack) {
        // Eliminate all rings in this stack
        totalRingsEliminated += stack.stackHeight;
        this.boardManager.removeStack(pos, this.gameState.board);
      }
    }
    
    // 5. Update elimination counts - ALL eliminated rings count toward moving player
    this.gameState.totalRingsEliminated += totalRingsEliminated;
    if (!this.gameState.board.eliminatedRings[movingPlayer]) {
      this.gameState.board.eliminatedRings[movingPlayer] = 0;
    }
    this.gameState.board.eliminatedRings[movingPlayer] += totalRingsEliminated;
    
    // Update player state
    this.updatePlayerEliminatedRings(movingPlayer, totalRingsEliminated);
    
    // 6. Mandatory self-elimination (one ring or cap from moving player)
    this.eliminatePlayerRingOrCap(movingPlayer);
  }

  /**
   * Calculate cap height for a ring stack
   * Rule Reference: Section 5.2 - Cap height is consecutive rings of same color from top
   */
  private calculateCapHeight(rings: number[]): number {
    if (rings.length === 0) return 0;
    
    const topColor = rings[0];
    let capHeight = 1;
    
    for (let i = 1; i < rings.length; i++) {
      if (rings[i] === topColor) {
        capHeight++;
      } else {
        break;
      }
    }
    
    return capHeight;
  }

  /**
   * Process markers along the movement path
   * Rule Reference: Section 8.3 - Marker Interaction
   */
  private processMarkersAlongPath(from: Position, to: Position, player: number): void {
    // Get all positions along the straight line path
    const path = this.getPathPositions(from, to);
    
    // Process each position in the path (excluding start and end)
    for (let i = 1; i < path.length - 1; i++) {
      const pos = path[i];
      const marker = this.boardManager.getMarker(pos, this.gameState.board);
      
      if (marker !== undefined) {
        if (marker === player) {
          // Own marker: collapse to territory (Section 8.3)
          this.boardManager.collapseMarker(pos, player, this.gameState.board);
        } else {
          // Opponent marker: flip to your color (Section 8.3)
          this.boardManager.flipMarker(pos, player, this.gameState.board);
        }
      }
    }
  }

  /**
   * Get all positions along a straight line path
   */
  private getPathPositions(from: Position, to: Position): Position[] {
    const path: Position[] = [from];
    
    // Calculate direction
    const dx = to.x - from.x;
    const dy = to.y - from.y;
    const dz = (to.z || 0) - (from.z || 0);
    
    // Normalize to step size of 1
    const steps = Math.max(Math.abs(dx), Math.abs(dy), Math.abs(dz));
    const stepX = steps > 0 ? dx / steps : 0;
    const stepY = steps > 0 ? dy / steps : 0;
    const stepZ = steps > 0 ? dz / steps : 0;
    
    // Generate all positions along the path
    for (let i = 1; i <= steps; i++) {
      const pos: Position = {
        x: Math.round(from.x + stepX * i),
        y: Math.round(from.y + stepY * i)
      };
      if (to.z !== undefined) {
        pos.z = Math.round((from.z || 0) + stepZ * i);
      }
      path.push(pos);
    }
    
    return path;
  }

  /**
   * Advance game through phases according to RingRift rules
   * Rule Reference: Section 4, Section 15.2
   * 
   * Phase Flow:
   * 1. ring_placement (optional unless no rings on board)
   * 2. movement (required if able)
   * 3. capture (optional to start, mandatory chaining)
   * 4. line_processing (automatic)
   * 5. territory_processing (automatic)
   * 6. Next player's turn
   */
  private advanceGame(): void {
    switch (this.gameState.currentPhase) {
      case 'ring_placement':
        // After placing a ring (or skipping), must move
        // Rule Reference: Section 4.1, 4.2
        this.gameState.currentPhase = 'movement';
        break;

      case 'movement':
        // After movement, check if captures are available
        // Rule Reference: Section 4.3
        const canCapture = this.hasValidCaptures(this.gameState.currentPlayer);
        if (canCapture) {
          this.gameState.currentPhase = 'capture';
        } else {
          // Skip to line processing
          this.gameState.currentPhase = 'line_processing';
        }
        break;

      case 'capture':
        // After captures complete, proceed to line processing
        // Rule Reference: Section 4.3, 4.5
        this.gameState.currentPhase = 'line_processing';
        break;

      case 'line_processing':
        // After processing lines, proceed to territory processing
        // Rule Reference: Section 4.5
        this.gameState.currentPhase = 'territory_processing';
        break;

      case 'territory_processing':
        // After processing territory, turn is complete
        // Check if player still has rings/stacks or needs to place
        // Rule Reference: Section 4, Section 4.1
        this.nextPlayer();
        
        // Determine starting phase for next player
        const playerStacks = this.boardManager.getPlayerStacks(this.gameState.board, this.gameState.currentPlayer);
        const currentPlayer = this.gameState.players.find(p => p.playerNumber === this.gameState.currentPlayer);
        
        // Rule Reference: Section 4.4 - Forced Elimination When Blocked
        // Check if player has no valid actions but controls stacks
        if (playerStacks.length > 0 && !this.hasValidActions(this.gameState.currentPlayer)) {
          // Player is blocked with stacks - must eliminate a cap
          this.processForcedElimination(this.gameState.currentPlayer);
          
          // After forced elimination, check victory conditions
          const gameEndCheck = this.ruleEngine.checkGameEnd(this.gameState);
          if (gameEndCheck.isGameOver) {
            // Game ended due to forced elimination
            this.endGame(gameEndCheck.winner, gameEndCheck.reason || 'forced_elimination');
            return; // Exit early - game is over
          }
          
          // Continue to next player after forced elimination
          this.nextPlayer();
          
          // Re-evaluate starting phase for the actual next player
          const nextPlayerStacks = this.boardManager.getPlayerStacks(this.gameState.board, this.gameState.currentPlayer);
          const nextPlayer = this.gameState.players.find(p => p.playerNumber === this.gameState.currentPlayer);
          
          if (nextPlayerStacks.length === 0 && nextPlayer && nextPlayer.ringsInHand > 0) {
            this.gameState.currentPhase = 'ring_placement';
          } else if (nextPlayer && nextPlayer.ringsInHand > 0) {
            this.gameState.currentPhase = 'ring_placement';
          } else {
            this.gameState.currentPhase = 'movement';
          }
        } else {
          // Normal turn progression
          if (playerStacks.length === 0 && currentPlayer && currentPlayer.ringsInHand > 0) {
            // No rings on board but has rings in hand - must place
            this.gameState.currentPhase = 'ring_placement';
          } else if (currentPlayer && currentPlayer.ringsInHand > 0) {
            // Has rings in hand and on board - can optionally place
            this.gameState.currentPhase = 'ring_placement';
          } else {
            // No rings in hand or all rings placed - go directly to movement
            this.gameState.currentPhase = 'movement';
          }
        }
        break;
    }
  }

  /**
   * Check if player has any valid capture moves available
   * Rule Reference: Section 10.1
   */
  private hasValidCaptures(playerNumber: number): boolean {
    const playerStacks = this.boardManager.getPlayerStacks(this.gameState.board, playerNumber);
    
    for (const stack of playerStacks) {
      // Check all adjacent positions for valid captures
      const adjacentPositions = this.getAdjacentPositions(stack.position);
      for (const adjPos of adjacentPositions) {
        const targetStack = this.boardManager.getStack(adjPos, this.gameState.board);
        if (targetStack && 
            targetStack.controllingPlayer !== playerNumber &&
            stack.capHeight >= targetStack.capHeight) {
          return true; // Found at least one valid capture
        }
      }
    }
    
    return false;
  }

  /**
   * Check if player has any valid placement moves
   * Rule Reference: Section 4.1, 6.1-6.3
   */
  private hasValidPlacements(playerNumber: number): boolean {
    const player = this.gameState.players.find(p => p.playerNumber === playerNumber);
    if (!player || player.ringsInHand === 0) {
      return false; // No rings in hand to place
    }

    // Check for any empty, non-collapsed spaces
    // For now, we'll do a simple check - in full implementation would check all positions
    // A player can place if they have rings in hand (placement restrictions like movement validation would be checked in the actual move)
    return true; // Simplified - assumes there's usually space to place
  }

  /**
   * Check if player has any valid movement moves
   * Rule Reference: Section 8.1, 8.2
   */
  private hasValidMovements(playerNumber: number): boolean {
    const playerStacks = this.boardManager.getPlayerStacks(this.gameState.board, playerNumber);
    
    if (playerStacks.length === 0) {
      return false; // No stacks to move
    }

    // For each player stack, check if it has any valid moves
    for (const stack of playerStacks) {
      const stackHeight = stack.stackHeight;
      
      // Check all 8 directions (or 6 for hexagonal)
      const directions = this.getAllDirections();
      
      for (const direction of directions) {
        // Check if we can move at least stack height in this direction
        let currentPos = stack.position;
        let distance = 0;
        let pathClear = true;
        
        for (let step = 1; step <= stackHeight + 5; step++) {
          const nextPos: Position = {
            x: stack.position.x + direction.x * step,
            y: stack.position.y + direction.y * step,
            ...(direction.z !== undefined && { z: (stack.position.z || 0) + direction.z * step })
          };
          
          if (!this.boardManager.isValidPosition(nextPos)) {
            break; // Out of bounds
          }
          
          // Check if this position is blocked (collapsed space or stack)
          if (this.boardManager.isCollapsedSpace(nextPos, this.gameState.board)) {
            break; // Blocked by collapsed space
          }
          
          const stackAtPos = this.boardManager.getStack(nextPos, this.gameState.board);
          if (stackAtPos) {
            break; // Blocked by another stack
          }
          
          // This position is reachable
          distance = step;
          
          // If we've met the minimum distance requirement, we have a valid move
          if (distance >= stackHeight) {
            return true;
          }
        }
      }
    }
    
    return false; // No valid movements found
  }

  /**
   * Get all movement directions based on board type
   */
  private getAllDirections(): { x: number; y: number; z?: number }[] {
    const config = BOARD_CONFIGS[this.gameState.boardType];
    
    if (config.type === 'hexagonal') {
      // Hexagonal directions (6 directions)
      return [
        { x: 1, y: 0, z: -1 },
        { x: 0, y: 1, z: -1 },
        { x: -1, y: 1, z: 0 },
        { x: -1, y: 0, z: 1 },
        { x: 0, y: -1, z: 1 },
        { x: 1, y: -1, z: 0 }
      ];
    } else {
      // Moore adjacency (8 directions) for square boards
      return [
        { x: 1, y: 0 },   // E
        { x: 1, y: 1 },   // SE
        { x: 0, y: 1 },   // S
        { x: -1, y: 1 },  // SW
        { x: -1, y: 0 },  // W
        { x: -1, y: -1 }, // NW
        { x: 0, y: -1 },  // N
        { x: 1, y: -1 }   // NE
      ];
    }
  }

  /**
   * Check if player has any valid actions available
   * Rule Reference: Section 4.4
   */
  private hasValidActions(playerNumber: number): boolean {
    return this.hasValidPlacements(playerNumber) || 
           this.hasValidMovements(playerNumber) || 
           this.hasValidCaptures(playerNumber);
  }

  /**
   * Force player to eliminate a cap when blocked with no valid moves
   * Rule Reference: Section 4.4 - Forced Elimination When Blocked
   */
  private processForcedElimination(playerNumber: number): void {
    const playerStacks = this.boardManager.getPlayerStacks(this.gameState.board, playerNumber);
    
    if (playerStacks.length === 0) {
      // No stacks to eliminate from - player forfeits turn
      return;
    }
    
    // TODO: In full implementation, player should choose which stack
    // For now, eliminate from first stack with a valid cap
    for (const stack of playerStacks) {
      if (stack.capHeight > 0) {
        // Found a stack with a cap, eliminate it
        this.eliminatePlayerRingOrCap(playerNumber);
        return;
      }
    }
  }

  /**
   * Get adjacent positions for a given position
   * Uses Moore adjacency (8-direction) for square boards, hexagonal for hex
   */
  private getAdjacentPositions(pos: Position): Position[] {
    const adjacent: Position[] = [];
    const config = BOARD_CONFIGS[this.gameState.boardType];
    
    if (config.type === 'hexagonal') {
      // Hexagonal adjacency (6 directions)
      const directions = [
        { x: 1, y: 0, z: -1 },
        { x: 0, y: 1, z: -1 },
        { x: -1, y: 1, z: 0 },
        { x: -1, y: 0, z: 1 },
        { x: 0, y: -1, z: 1 },
        { x: 1, y: -1, z: 0 }
      ];
      
      for (const dir of directions) {
        const newPos: Position = {
          x: pos.x + dir.x,
          y: pos.y + dir.y,
          z: (pos.z || 0) + dir.z
        };
        if (this.boardManager.isValidPosition(newPos)) {
          adjacent.push(newPos);
        }
      }
    } else {
      // Moore adjacency for square boards (8 directions)
      for (let dx = -1; dx <= 1; dx++) {
        for (let dy = -1; dy <= 1; dy++) {
          if (dx === 0 && dy === 0) continue;
          
          const newPos: Position = {
            x: pos.x + dx,
            y: pos.y + dy
          };
          if (this.boardManager.isValidPosition(newPos)) {
            adjacent.push(newPos);
          }
        }
      }
    }
    
    return adjacent;
  }

  private nextPlayer(): void {
    const currentIndex = this.gameState.players.findIndex(p => p.playerNumber === this.gameState.currentPlayer);
    const nextIndex = (currentIndex + 1) % this.gameState.players.length;
    this.gameState.currentPlayer = this.gameState.players[nextIndex].playerNumber;
  }

  private startPlayerTimer(playerNumber: number): void {
    const player = this.gameState.players.find(p => p.playerNumber === playerNumber);
    if (!player || player.type === 'ai') return;

    const timer = setTimeout(() => {
      // Time expired, forfeit the game
      this.forfeitGame(playerNumber.toString());
    }, player.timeRemaining);

    this.moveTimers.set(playerNumber, timer);
  }

  private stopPlayerTimer(playerNumber: number): void {
    const timer = this.moveTimers.get(playerNumber);
    if (timer) {
      clearTimeout(timer);
      this.moveTimers.delete(playerNumber);
    }
  }

  private endGame(winner?: number, reason?: string): {
    success: boolean;
    gameResult: GameResult;
  } {
    this.gameState.gameStatus = 'completed';
    this.gameState.winner = winner;

    // Clear all timers
    for (const timer of this.moveTimers.values()) {
      clearTimeout(timer);
    }
    this.moveTimers.clear();

    // Calculate final scores
    const finalScore: { [playerNumber: number]: number } = {};
    for (const player of this.gameState.players) {
      const playerStacks = this.boardManager.getPlayerStacks(this.gameState.board, player.playerNumber);
      const stackCount = playerStacks.reduce((sum, stack) => sum + stack.stackHeight, 0);
      
      const territories = this.boardManager.findPlayerTerritories(this.gameState.board, player.playerNumber);
      const territorySize = territories.reduce((sum, territory) => sum + territory.spaces.length, 0);
      
      finalScore[player.playerNumber] = stackCount + territorySize;
    }

    const gameResult: GameResult = {
      ...(winner !== undefined && { winner }),
      reason: (reason as any) || 'game_completed',
      finalScore: {
        ringsEliminated: {},
        territorySpaces: {},
        ringsRemaining: finalScore
      },
    };

    // Update player ratings if this is a rated game
    if (this.gameState.isRated) {
      this.updatePlayerRatings(gameResult);
    }

    return {
      success: true,
      gameResult
    };
  }

  private updatePlayerRatings(gameResult: GameResult): void {
    // Rating calculation logic would go here
    const winnerPlayer = this.gameState.players.find(p => p.playerNumber === gameResult.winner);
    const loserPlayers = this.gameState.players.filter(p => p.playerNumber !== gameResult.winner);

    // For now, just log the rating update
    console.log('Rating update needed for:', {
      winner: winnerPlayer?.username,
      losers: loserPlayers.map(p => p.username)
    });
  }

  addSpectator(userId: string): boolean {
    if (!this.gameState.spectators.includes(userId)) {
      this.gameState.spectators.push(userId);
      return true;
    }
    return false;
  }

  removeSpectator(userId: string): boolean {
    const index = this.gameState.spectators.indexOf(userId);
    if (index !== -1) {
      this.gameState.spectators.splice(index, 1);
      return true;
    }
    return false;
  }

  pauseGame(): boolean {
    if (this.gameState.gameStatus === 'active') {
      this.gameState.gameStatus = 'paused';
      
      // Stop current player's timer
      this.stopPlayerTimer(this.gameState.currentPlayer);
      
      return true;
    }
    return false;
  }

  resumeGame(): boolean {
    if (this.gameState.gameStatus === 'paused') {
      this.gameState.gameStatus = 'active';
      
      // Restart current player's timer
      this.startPlayerTimer(this.gameState.currentPlayer);
      
      return true;
    }
    return false;
  }

  forfeitGame(playerNumber: string): {
    success: boolean;
    gameResult?: GameResult;
  } {
    const winner = this.gameState.players.find(p => p.playerNumber !== parseInt(playerNumber))?.playerNumber;
    
    return this.endGame(winner, 'resignation');
  }

  getValidMoves(_playerNumber: number): Move[] {
    // This would return all valid moves for the current player
    // For now, return empty array
    return [];
  }

}

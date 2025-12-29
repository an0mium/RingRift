# RingRift WebSocket API

> **Doc Status (2025-12-29): Active**
>
> **SSoT:** `src/shared/types/websocket.ts` (event names + payload types) and
> `src/shared/validation/websocketSchemas.ts` (runtime validation).
>
> **Related:** `docs/architecture/API_REFERENCE.md`,
> `docs/architecture/PLAYER_MOVE_TRANSPORT_DECISION.md`,
> `src/server/websocket/server.ts`.

## Overview

- Transport: Socket.IO on the same host/port as the HTTP API.
- Authentication: JWT provided as `socket.handshake.auth.token` or `?token=...`
  in the connection query string (see `src/server/websocket/server.ts`).
- Game rooms: clients join a game-specific room via `join_game`.
- Lobby stream: clients subscribe via `lobby:subscribe` for lobby broadcasts.

This document is a summary; the authoritative contract is the TypeScript
types and Zod schemas listed above.

## Event Contract (Summary)

### Server to Client events

- `game_state` - snapshot updates during play.
- `game_over` - terminal game result and metadata.
- `game_error` - fatal game-level error that ends a game.
- `player_joined`, `player_left`, `player_disconnected`, `player_reconnected`.
- `chat_message`, `chat_message_persisted`, `chat_history`.
- `player_choice_required`, `player_choice_canceled`.
- `decision_phase_timeout_warning`, `decision_phase_timed_out`.
- `rematch_requested`, `rematch_response`.
- `position_evaluation` (optional, analysis mode).
- `time_update` (legacy/experimental).
- `error` - transport-level error payloads.
- `diagnostic:pong` - load-test ping response.
- `lobby:game_created`, `lobby:game_joined`, `lobby:game_started`, `lobby:game_cancelled`.
- `match-found`, `matchmaking-status`.
- `request_reconnect` (reserved, optional).

### Client to Server events

- `join_game`, `leave_game`.
- `player_move`, `player_move_by_id`.
- `player_choice_response`.
- `chat_message`.
- `rematch_request`, `rematch_respond`.
- `diagnostic:ping` (load testing).
- `lobby:subscribe`, `lobby:unsubscribe`.
- `matchmaking:join`, `matchmaking:leave`.

## Payload Types

- **Event names + payload interfaces:** `src/shared/types/websocket.ts`.
- **Runtime validation:** `src/shared/validation/websocketSchemas.ts`.
- **Shared game types:** `src/shared/types/game.ts`.

If you add or change an event, update the TypeScript types and Zod schemas
first, then reflect the change here.

## Error Semantics

- `error`: transport-level failures (validation, auth, rate limiting). See
  `WebSocketErrorPayload` in `src/shared/types/websocket.ts`.
- `game_error`: game-level terminal errors (AI failure, rules service failure).

## Minimal Client Example

```ts
import { io } from 'socket.io-client';

const socket = io('http://localhost:3000', {
  auth: { token: '<jwt>' },
});

socket.emit('join_game', { gameId: 'abc123' });
socket.on('game_state', (payload) => {
  console.log('state', payload);
});
```

"""WebSocket broadcast manager."""

from __future__ import annotations

import asyncio
from typing import Any


class WebSocketManager:
    def __init__(self) -> None:
        self.active_connections: list[Any] = []
        self.broadcast_history: list[dict[str, Any]] = []
        self._connection_lock = asyncio.Lock()

    async def connect(
        self,
        websocket: Any,
        *,
        max_connections: int | None = None,
        subprotocol: str | None = None,
    ) -> bool:
        async with self._connection_lock:
            if max_connections is not None:
                if len(self.active_connections) >= max_connections:
                    return False
            await _accept_websocket(websocket, subprotocol)
            self.active_connections.append(websocket)
            return True

    def disconnect(self, websocket: Any) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict[str, Any]) -> None:
        self.broadcast_history.append(message)
        for websocket in list(self.active_connections):
            sender = getattr(websocket, "send_json", None)
            if sender is not None:
                await sender(message)


async def _accept_websocket(websocket: Any, subprotocol: str | None) -> None:
    accept = getattr(websocket, "accept", None)
    if accept is None:
        return
    if subprotocol is None:
        await accept()
        return
    try:
        await accept(subprotocol=subprotocol)
    except TypeError:
        await accept()


__all__ = ["WebSocketManager"]

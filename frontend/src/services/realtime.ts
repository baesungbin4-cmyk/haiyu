import { useEffect, useState } from "react";

export type ConnectionStatus = "online" | "degraded" | "offline";

export interface RealtimeStatus {
  edge: ConnectionStatus;
  lastEventType: string;
  lastMessage: RealtimeMessage | null;
  mqtt: ConnectionStatus;
  ws: ConnectionStatus;
}

export type RealtimeMessage = Record<string, unknown>;

const BASE_PROTOCOL = "haiyu.realtime.v1";
const TOKEN_PREFIX = "bearer.";
const DEFAULT_WS_BASE_URL = "ws://localhost:8000";
const STALE_TIMEOUT_MS = 15_000;
const RECONNECT_DELAY_MS = 5_000;

const INITIAL_STATUS: RealtimeStatus = {
  edge: "degraded",
  lastEventType: "idle",
  lastMessage: null,
  mqtt: "degraded",
  ws: "degraded",
};

export function useRealtimeStatus() {
  const [status, setStatus] = useState<RealtimeStatus>(INITIAL_STATUS);

  useEffect(() => {
    let socket: WebSocket | null = null;
    let reconnectTimer: number | undefined;
    let staleTimer: number | undefined;
    let disposed = false;

    const markStale = () => {
      setStatus((current) => ({
        ...current,
        edge: current.ws === "online" ? "degraded" : current.edge,
        mqtt: current.ws === "online" ? "degraded" : current.mqtt,
      }));
    };

    const resetStaleTimer = () => {
      if (staleTimer !== undefined) {
        window.clearTimeout(staleTimer);
      }
      staleTimer = window.setTimeout(markStale, STALE_TIMEOUT_MS);
    };

    const connect = () => {
      if (disposed) {
        return;
      }
      setStatus((current) => ({ ...current, ws: "degraded" }));
      socket = new WebSocket(realtimeUrl(), websocketProtocols());

      socket.onopen = () => {
        if (disposed) {
          return;
        }
        setStatus((current) => ({ ...current, ws: "online" }));
        resetStaleTimer();
      };
      socket.onmessage = (event) => {
        if (disposed) {
          return;
        }
        const message = parseRealtimeMessage(event.data);
        if (message === null) {
          return;
        }
        setStatus((current) => statusFromMessage(current, message));
        resetStaleTimer();
      };
      socket.onerror = () => {
        if (disposed) {
          return;
        }
        setStatus((current) => ({ ...current, ws: "degraded" }));
      };
      socket.onclose = () => {
        if (disposed) {
          return;
        }
        if (staleTimer !== undefined) {
          window.clearTimeout(staleTimer);
        }
        setStatus((current) => ({
          ...current,
          edge: "degraded",
          mqtt: "degraded",
          ws: "offline",
        }));
        if (!disposed) {
          reconnectTimer = window.setTimeout(connect, RECONNECT_DELAY_MS);
        }
      };
    };

    connect();

    return () => {
      disposed = true;
      if (reconnectTimer !== undefined) {
        window.clearTimeout(reconnectTimer);
      }
      if (staleTimer !== undefined) {
        window.clearTimeout(staleTimer);
      }
      socket?.close();
    };
  }, []);

  return status;
}

function realtimeUrl() {
  const baseUrl = import.meta.env.VITE_WS_BASE_URL || DEFAULT_WS_BASE_URL;
  return `${baseUrl.replace(/\/$/, "")}/ws/realtime`;
}

function websocketProtocols() {
  const protocols = [BASE_PROTOCOL];
  const token = window.localStorage.getItem("apiToken");
  if (!token) {
    return protocols;
  }
  protocols.push(`${TOKEN_PREFIX}${base64UrlEncode(token)}`);
  return protocols;
}

function base64UrlEncode(value: string) {
  const bytes = new TextEncoder().encode(value);
  let binary = "";
  bytes.forEach((byte) => {
    binary += String.fromCharCode(byte);
  });
  return window
    .btoa(binary)
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/g, "");
}

function parseRealtimeMessage(value: string) {
  try {
    const parsed: unknown = JSON.parse(value);
    if (isRecord(parsed)) {
      return parsed;
    }
  } catch {
    return null;
  }
  return null;
}

function statusFromMessage(
  current: RealtimeStatus,
  message: Record<string, unknown>,
): RealtimeStatus {
  const type = eventType(message);
  const data = eventData(message);
  if (type === "alert" || type === "alarm" || type === "track") {
    return {
      edge: "online",
      lastEventType: type,
      lastMessage: message,
      mqtt: "online",
      ws: "online",
    };
  }
  if (type === "status" || type === "lwt") {
    const edgeOnline = type === "lwt" ? false : onlineValue(data);
      return {
        edge: edgeOnline ? "online" : "offline",
        lastEventType: type,
        lastMessage: message,
        mqtt: "online",
        ws: "online",
      };
    }
  return {
    ...current,
    lastEventType: type || "message",
    lastMessage: message,
    ws: "online",
  };
}

function eventType(message: Record<string, unknown>) {
  const type =
    stringValue(message.type) ||
    stringValue(message.event_type) ||
    stringValue(message.eventType);
  return type.toLowerCase();
}

function eventData(message: Record<string, unknown>) {
  if (isRecord(message.data)) {
    return message.data;
  }
  if (isRecord(message.payload)) {
    return message.payload;
  }
  return message;
}

function onlineValue(data: Record<string, unknown>) {
  const value = data.online ?? data.status ?? data.state;
  if (typeof value === "boolean") {
    return value;
  }
  const normalized = stringValue(value).toLowerCase();
  return ["1", "ok", "online", "true"].includes(normalized);
}

function stringValue(value: unknown) {
  if (typeof value === "string") {
    return value.trim();
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return "";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

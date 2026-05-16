export function chatSessionId(): string {
  if (typeof window === "undefined") return "server-session";

  const stored = window.localStorage.getItem("persona-chat-session-id");
  if (stored) return stored;

  const next = crypto.randomUUID();
  window.localStorage.setItem("persona-chat-session-id", next);
  return next;
}

export function resetChatSessionId(): string {
  if (typeof window === "undefined") return "server-session";

  const next = crypto.randomUUID();
  window.localStorage.setItem("persona-chat-session-id", next);
  return next;
}

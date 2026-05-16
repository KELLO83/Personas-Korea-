import { NODE_TYPE_LABELS, RELATION_LABELS } from "./constants";

export function compactNumber(value: number): string {
  return new Intl.NumberFormat("ko-KR", { notation: "compact", maximumFractionDigits: 1 }).format(value);
}

export function fullNumber(value: number): string {
  return new Intl.NumberFormat("ko-KR").format(value);
}

export function percent(value: number | null | undefined): string {
  if (value === null || value === undefined) return "-";
  return `${(value * 100).toFixed(1)}%`;
}

export function shortUuid(uuid: string): string {
  return uuid.length > 10 ? `${uuid.slice(0, 8)}…` : uuid;
}

export function uuidWithName(uuid: string, name: string | null | undefined): string {
  const cleanName = name?.trim();
  if (!cleanName || cleanName === uuid || cleanName === shortUuid(uuid) || cleanName === "기본 페르소나") return uuid;
  return `${uuid}(${cleanName})`;
}

export function nodeTypeLabel(value: string): string {
  return NODE_TYPE_LABELS[value] ?? value;
}

export function relationLabel(value: string): string {
  return RELATION_LABELS[value] ?? value;
}

export function joinDefined(values: Array<string | number | null | undefined>, fallback = "정보 없음"): string {
  const result = values.filter((value) => value !== null && value !== undefined && String(value).trim()).join(" · ");
  return result || fallback;
}

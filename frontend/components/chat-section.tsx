import { FormEvent } from "react";
import type { ChatMessage } from "@/lib/api-types";

interface ChatSectionProps {
  messages: ChatMessage[];
  input: string;
  loading: boolean;
  error: string | null;
  onInputChange: (value: string) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => Promise<void>;
  onReset: () => void;
}

export function ChatSection({ messages, input, loading, error, onInputChange, onSubmit, onReset }: ChatSectionProps) {
  return (
    <section className="grid two">
      <div className="card">
        <h2>대화형 탐색</h2>
        <p className="muted">예: “서울 30대 여성들이 많이 가진 취미는?”처럼 자연어로 질문하세요. 이전 질문의 조건은 같은 대화 안에서 이어집니다.</p>
        <form onSubmit={(event) => void onSubmit(event)}>
          <textarea className="textarea" value={input} onChange={(event) => onInputChange(event.target.value)} placeholder="질문을 입력하세요" />
          <div className="card-actions">
            <button className="primary-button" disabled={loading}>{loading ? "분석 중…" : "질문 보내기"}</button>
            <button className="ghost-button" type="button" onClick={onReset} disabled={loading && messages.length === 0}>새 대화</button>
          </div>
        </form>
        {error && <div className="card error-box" style={{ marginTop: 12 }}>{error}</div>}
      </div>
      <div className="card">
        <h3>대화 기록</h3>
        <div className="message-list">
          {messages.length === 0 && <p className="muted">아직 대화가 없습니다.</p>}
          {messages.map((message, index) => (
            <div className={`message ${message.role}`} key={`${message.role}-${index}`}>
              <strong>{message.role === "user" ? "사용자" : "Nemotron"}</strong>
              <p>{message.content}</p>
              {message.filters && Object.keys(message.filters).length > 0 && (
                <div className="pill-row">{Object.entries(message.filters).map(([key, value]) => <span className="pill" key={key}>{key}: {value}</span>)}</div>
              )}
              {message.sources && message.sources.length > 0 && (
                <details className="small muted">
                  <summary>검색 근거 {message.sources.length}건</summary>
                  <pre>{JSON.stringify(message.sources, null, 2)}</pre>
                </details>
              )}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

import { useEffect, useRef, useState } from "react";
import { askQuestion, getSyncStatus, syncDocuments } from "./api";
import "./App.css";

function Message({ msg, onRetry }) {
  if (msg.role === "user") {
    return <div className="msg user">{msg.text}</div>;
  }
  if (msg.error) {
    return (
      <div className="msg bot error">
        <p>{msg.error}</p>
        <button onClick={() => onRetry(msg.question)}>Retry</button>
      </div>
    );
  }
  return (
    <div className={"msg bot" + (msg.found ? "" : " notfound")}>
      <p>{msg.text}</p>
      {msg.sources?.length > 0 && (
        <div className="sources">
          Sources:{" "}
          {msg.sources.map((s) => (
            <a key={s.url} href={s.url} target="_blank" rel="noreferrer">
              {s.title}
            </a>
          ))}
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [syncMsg, setSyncMsg] = useState("");
  const [syncing, setSyncing] = useState(false);
  const [status, setStatus] = useState(null);
  const bottomRef = useRef(null);

  const refreshStatus = () => getSyncStatus().then(setStatus).catch(() => setStatus(null));
  useEffect(() => { refreshStatus(); }, []);
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, loading]);

  async function ask(question) {
    setLoading(true);
    try {
      const res = await askQuestion(question);
      setMessages((m) => [...m, { role: "bot", text: res.answer, found: res.found, sources: res.sources }]);
    } catch (err) {
      setMessages((m) => [...m, { role: "bot", error: err.message, question }]);
    } finally {
      setLoading(false);
    }
  }

  function send(e) {
    e.preventDefault();
    const q = input.trim();
    if (!q || loading) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", text: q }]);
    ask(q);
  }

  function retry(question) {
    setMessages((m) => m.filter((x) => !(x.error && x.question === question)));
    ask(question);
  }

  async function sync() {
    setSyncing(true);
    setSyncMsg("Syncing documents… the first run downloads the embedding model and may take a while.");
    try {
      const r = await syncDocuments();
      setSyncMsg(
        `Done: ${r.added} added, ${r.updated} updated, ${r.unchanged} unchanged, ${r.removed} removed` +
          (r.skipped ? `, ${r.skipped} empty` : "") +
          (r.errors.length ? `. ${r.errors.length} document(s) could not be read.` : ".")
      );
      refreshStatus();
    } catch (err) {
      setSyncMsg("Sync failed: " + err.message);
    } finally {
      setSyncing(false);
    }
  }

  return (
    <main className="app">
      <header>
        <h1>Drive Chatbot</h1>
        <div className="sync">
          <button onClick={sync} disabled={syncing}>{syncing ? "Syncing…" : "Sync documents"}</button>
          <small>
            {status?.last_sync
              ? `Last sync: ${new Date(status.last_sync).toLocaleString()} (${status.indexed_documents} docs)`
              : "Not synced yet"}
          </small>
        </div>
        {syncMsg && <small className="syncmsg">{syncMsg}</small>}
      </header>

      <section className="chat">
        {messages.length === 0 && <p className="hint">Sync your documents, then ask a question about them.</p>}
        {messages.map((m, i) => <Message key={i} msg={m} onRetry={retry} />)}
        {loading && <div className="msg bot loading">Thinking…</div>}
        <div ref={bottomRef} />
      </section>

      <form onSubmit={send}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about your documents…"
          maxLength={2000}
          disabled={loading}
        />
        <button type="submit" disabled={loading || !input.trim()}>Send</button>
      </form>
    </main>
  );
}

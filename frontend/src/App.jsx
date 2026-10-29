import { useEffect, useRef, useState } from "react";
import { askQuestion, getSyncStatus, syncDocuments } from "./api";
import logoUrl from "./assets/appinventiv-logo.png";
import Sidebar, { Logo } from "./components/Sidebar";
import Message, { Thinking } from "./components/Message";
import Composer from "./components/Composer";
import { MenuIcon, SyncIcon } from "./components/icons";

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [syncMsg, setSyncMsg] = useState("");
  const [syncing, setSyncing] = useState(false);
  const [status, setStatus] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
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

  const statusText = status?.last_sync
    ? `Last sync: ${new Date(status.last_sync).toLocaleString()} (${status.indexed_documents} docs)`
    : "Not synced yet";

  return (
    <div className="flex h-full overflow-hidden">
      <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />

      <main className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center gap-3 px-4 pb-2 pt-4 sm:px-8 sm:pt-6">
          <button
            onClick={() => setSidebarOpen(true)}
            className="rounded-full bg-white p-2 text-gray-700 shadow-card lg:hidden"
            aria-label="Open sidebar"
          >
            <MenuIcon />
          </button>
          <div className="min-w-0 flex-1">
            <h1 className="truncate text-lg font-semibold sm:text-xl">
              <span className="lg:hidden"><Logo /></span>
              <span className="hidden lg:inline">Drive Chatbot</span>
            </h1>
            <p className="hidden truncate text-xs text-muted sm:block">{statusText}</p>
          </div>
          <button
            onClick={sync}
            disabled={syncing}
            className="flex items-center gap-2 rounded-full bg-brand-500 px-4 py-2 text-sm font-medium text-white transition hover:bg-brand-600 disabled:cursor-not-allowed disabled:bg-brand-100"
          >
            <SyncIcon className={`h-4 w-4 ${syncing ? "animate-spin" : ""}`} />
            <span className="hidden sm:inline">{syncing ? "Syncing…" : "Sync documents"}</span>
          </button>
        </header>

        <div className="px-4 sm:px-8">
          <p className="truncate text-xs text-muted sm:hidden">{statusText}</p>
          {syncMsg && (
            <p className="mt-2 rounded-2xl bg-white px-4 py-2 text-xs text-gray-600 shadow-card">{syncMsg}</p>
          )}
        </div>

        <section className="flex-1 overflow-y-auto px-4 py-6 sm:px-8">
          <div className="mx-auto flex min-h-full max-w-3xl flex-col gap-5">
            {messages.length === 0 && (
              <div className="my-auto text-center">
                <img
                  src={logoUrl}
                  alt="Appinventiv"
                  className="mx-auto mb-5 h-28 w-28 rounded-3xl shadow-card sm:h-32 sm:w-32"
                />
                <h2 className="mb-1 text-xl font-semibold">Appinventiv</h2>
                <p className="text-muted">Sync your documents, then ask a question about them.</p>
              </div>
            )}
            {messages.map((m, i) => <Message key={i} msg={m} onRetry={retry} />)}
            {loading && <Thinking />}
            <div ref={bottomRef} />
          </div>
        </section>

        <div className="px-4 pb-4 sm:px-8 sm:pb-6">
          <Composer value={input} onChange={setInput} onSubmit={send} loading={loading} />
        </div>
      </main>
    </div>
  );
}

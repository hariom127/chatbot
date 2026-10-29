import { DocIcon, RetryIcon } from "./icons";
import Markdown from "./Markdown";

function BotAvatar() {
  return (
    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-brand-500 text-[11px] font-bold text-white">
      AI
    </div>
  );
}

export function Thinking() {
  return (
    <div className="flex gap-3">
      <BotAvatar />
      <div className="flex items-center gap-1.5 rounded-2xl rounded-tl-md bg-white px-4 py-3 text-sm text-muted shadow-card">
        <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-brand-500 [animation-delay:-0.3s]" />
        <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-brand-500 [animation-delay:-0.15s]" />
        <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-brand-500" />
        <span className="ml-1.5 italic">Thinking…</span>
      </div>
    </div>
  );
}

export default function Message({ msg, onRetry }) {
  if (msg.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-tr-md bg-brand-500 px-4 py-3 text-[15px] leading-relaxed text-white shadow-sm">
          {msg.text}
        </div>
      </div>
    );
  }

  if (msg.error) {
    return (
      <div className="flex gap-3">
        <BotAvatar />
        <div className="max-w-[85%] rounded-2xl rounded-tl-md border border-red-200 bg-red-50 px-4 py-3 text-[15px] text-red-800">
          <p className="whitespace-pre-wrap">{msg.error}</p>
          <button
            onClick={() => onRetry(msg.question)}
            className="mt-3 inline-flex items-center gap-1.5 rounded-full bg-white px-3 py-1.5 text-sm font-medium text-red-700 ring-1 ring-red-200 transition hover:bg-red-100"
          >
            <RetryIcon className="h-4 w-4" /> Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex gap-3">
      <BotAvatar />
      <div
        className={`max-w-[85%] rounded-2xl rounded-tl-md px-4 py-3 text-[15px] leading-relaxed shadow-card
          ${msg.found ? "bg-white text-gray-800" : "border border-amber-200 bg-amber-50 text-amber-900"}`}
      >
        <Markdown text={msg.text} />
        {msg.sources?.length > 0 && (
          <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-gray-100 pt-3 text-xs">
            <span className="font-medium text-muted">Sources:</span>
            {msg.sources.map((s) => (
              <a
                key={s.url}
                href={s.url}
                target="_blank"
                rel="noreferrer"
                className="inline-flex max-w-full items-center gap-1 rounded-full bg-brand-50 px-2.5 py-1 font-medium text-brand-600 transition hover:bg-brand-100"
              >
                <DocIcon className="h-3.5 w-3.5 shrink-0" />
                <span className="truncate">{s.title}</span>
              </a>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

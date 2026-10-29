import { SendIcon } from "./icons";

export default function Composer({ value, onChange, onSubmit, loading }) {
  return (
    <form onSubmit={onSubmit} className="mx-auto flex w-full max-w-3xl items-center gap-2 rounded-full bg-white p-2 pl-5 shadow-card ring-1 ring-gray-100 focus-within:ring-brand-500">
      <input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Ask about your documents…"
        maxLength={2000}
        disabled={loading}
        className="min-w-0 flex-1 bg-transparent py-2 text-[15px] outline-none placeholder:text-gray-400 disabled:cursor-not-allowed"
      />
      <button
        type="submit"
        disabled={loading || !value.trim()}
        className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-brand-500 text-white transition hover:bg-brand-600 disabled:cursor-not-allowed disabled:bg-brand-100"
        aria-label="Send"
      >
        <SendIcon className="h-5 w-5" />
      </button>
    </form>
  );
}

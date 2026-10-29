import logoUrl from "../assets/appinventiv-logo.png";
import { ChatIcon, CloseIcon, PlusIcon, SearchIcon, TrashIcon } from "./icons";

// Placeholder data: chat history is not implemented yet, so the sidebar is static UI only.
const DUMMY_GROUPS = [
  {
    label: "Today",
    items: ["Leave policy questions", "Onboarding checklist", "Expense reimbursement steps"],
  },
  {
    label: "Previous 7 days",
    items: ["Remote work guidelines", "Q3 project timeline", "Security training summary"],
  },
];

export function Logo() {
  return (
    <span className="inline-flex items-center gap-2 text-xl font-bold tracking-tight">
      <img src={logoUrl} alt="" className="h-8 w-8 rounded-lg" />
      Appinventiv
    </span>
  );
}

export default function Sidebar({ open, onClose }) {
  return (
    <>
      {/* Mobile backdrop */}
      <div
        className={`fixed inset-0 z-30 bg-gray-900/40 transition-opacity lg:hidden ${open ? "opacity-100" : "pointer-events-none opacity-0"}`}
        onClick={onClose}
      />
      <aside
        className={`fixed inset-y-0 left-0 z-40 flex w-[300px] max-w-[85vw] flex-col bg-white p-5 shadow-card transition-transform
          lg:static lg:z-auto lg:m-4 lg:mr-0 lg:max-w-none lg:translate-x-0 lg:rounded-3xl
          ${open ? "translate-x-0" : "-translate-x-full"}`}
      >
        <div className="flex items-center justify-between pb-6 pt-1">
          <Logo />
          <button onClick={onClose} className="rounded-full p-1.5 text-muted hover:bg-gray-100 lg:hidden" aria-label="Close sidebar">
            <CloseIcon />
          </button>
        </div>

        <div className="flex gap-2">
          <button
            type="button"
            className="flex flex-1 items-center justify-center gap-2 rounded-full bg-brand-500 px-4 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-brand-600"
          >
            <PlusIcon className="h-4 w-4" /> New chat
          </button>
          <button
            type="button"
            className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-brand-50 text-ink transition hover:bg-brand-100"
            aria-label="Search conversations"
          >
            <SearchIcon className="h-4 w-4" />
          </button>
        </div>

        <div className="mt-6 flex items-center justify-between border-b border-gray-100 pb-3 text-xs">
          <span className="font-medium text-muted">Your conversations</span>
          <button type="button" className="font-semibold text-brand-500 hover:text-brand-700">
            Clear All
          </button>
        </div>

        <nav className="-mx-2 mt-2 flex-1 overflow-y-auto px-2">
          {DUMMY_GROUPS.map((g) => (
            <div key={g.label} className="mb-3">
              <p className="px-2 pb-1 pt-2 text-[11px] font-medium uppercase tracking-wide text-gray-400">{g.label}</p>
              <ul className="space-y-0.5">
                {g.items.map((title, i) => {
                  const active = g === DUMMY_GROUPS[0] && i === 0;
                  return (
                    <li key={title} className="group relative">
                      <button
                        type="button"
                        className={`flex w-full items-center gap-3 rounded-full py-2.5 pl-3 pr-10 text-left text-sm transition
                          ${active ? "bg-brand-50 font-semibold text-brand-600" : "text-gray-700 hover:bg-gray-50"}`}
                      >
                        <ChatIcon className="h-4 w-4 shrink-0" />
                        <span className="truncate">{title}</span>
                      </button>
                      <button
                        type="button"
                        className={`absolute right-2 top-1/2 -translate-y-1/2 rounded-full p-1.5 text-gray-400 transition hover:bg-white hover:text-red-500
                          ${active ? "opacity-100" : "opacity-100 lg:opacity-0 lg:group-hover:opacity-100"}`}
                        aria-label={`Delete conversation ${title}`}
                      >
                        <TrashIcon className="h-4 w-4" />
                      </button>
                    </li>
                  );
                })}
              </ul>
            </div>
          ))}
        </nav>
      </aside>
    </>
  );
}

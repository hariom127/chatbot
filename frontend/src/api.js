const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function request(path, options) {
  let res;
  try {
    res = await fetch(`${API_BASE_URL}${path}`, options);
  } catch {
    throw new Error("Cannot reach the backend. Is it running on " + API_BASE_URL + "?");
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = typeof data.detail === "string" ? data.detail : "Request failed (" + res.status + ")";
    throw new Error(detail);
  }
  return data;
}

export const askQuestion = (question, history = []) =>
  request("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, history }),
  });

export const syncDocuments = () => request("/sync", { method: "POST" });
export const getSyncStatus = () => request("/sync/status");

export class ApiError extends Error {
  constructor(status, message) {
    super(message || `HTTP ${status}`);
    this.status = status;
  }
}

export async function request(path, { method = "GET", body } = {}) {
  const res = await fetch(path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  let data = null;
  try {
    data = await res.json();
  } catch {
    throw new ApiError(res.status, "invalid JSON response");
  }
  // New contract: {ok, data} / {ok:false, error}. Tolerate legacy flat shape.
  if (!res.ok || data.ok === false) {
    throw new ApiError(res.status, data.error || `HTTP ${res.status}`);
  }
  return data.data !== undefined ? data.data : data;
}

export const get = (path) => request(path);
export const post = (path, body = {}) => request(path, { method: "POST", body });
export const put = (path, body = {}) => request(path, { method: "PUT", body });
export const del = (path) => request(path, { method: "DELETE" });

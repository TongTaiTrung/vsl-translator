import { ApiError } from "./client.js";

export async function postFrame(blob) {
  const res = await fetch("/api/frames", {
    method: "POST",
    headers: { "Content-Type": "image/jpeg" },
    body: blob,
  });
  let data = null;
  try {
    data = await res.json();
  } catch {
    throw new ApiError(res.status, "Phản hồi không phải JSON");
  }
  if (!res.ok || data.ok === false) {
    throw new ApiError(res.status, data.error || `HTTP ${res.status}`);
  }
  return data.data !== undefined ? data.data : data;
}

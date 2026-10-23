import { put } from "./client.js";

export const setGroqKey = (key) => put("/api/config/groq-key", { key });

import { post } from "./client.js";

export const retrySmart = (sentence) =>
  post("/api/smart/retries", { sentence });

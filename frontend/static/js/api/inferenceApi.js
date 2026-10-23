import { post } from "./client.js";

export const startInference = (debug = false) =>
  post("/api/inference/start", { debug });
export const stopInference = () => post("/api/inference/stop");

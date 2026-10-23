import { post, get } from "./client.js";

export const startMic = () => post("/api/stt/start");
export const stopMic = () => post("/api/stt/stop");
export const pollStt = () => get("/api/stt");

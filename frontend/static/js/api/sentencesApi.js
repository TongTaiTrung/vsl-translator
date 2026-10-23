import { get, del } from "./client.js";

export const listSentences = () => get("/api/sentences");
export const clearSentence = () => del("/api/sentences");
export const backspaceWord = () => del("/api/sentences/last");

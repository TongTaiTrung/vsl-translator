import { get } from "./client.js";

export const getState = () => get("/api/state");

export function pollState(intervalMs, onUpdate) {
  const tick = async () => {
    try {
      onUpdate(await getState());
    } catch {
      /* server/camera chưa sẵn sàng */
    }
  };
  tick();
  return setInterval(tick, intervalMs);
}

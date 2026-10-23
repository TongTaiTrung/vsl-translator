const listeners = new Set();

export const store = {
  state: {
    status: "...",
    buffer: "-",
    pred: null,
    conf: 0,
    sentence: [],
    smart: "",
    smart_pending: false,
    smart_error: null,
    groq_model: "",
    groq_configured: false,
    inferring: false,
  },
  set(patch) {
    if (patch && Number.isFinite(patch.rev) && Number.isFinite(this.state.rev)
        && patch.rev < this.state.rev) {
      return;
    }
    Object.assign(this.state, patch);
    for (const fn of listeners) fn(this.state);
  },
  subscribe(fn) {
    listeners.add(fn);
    return () => listeners.delete(fn);
  },
};

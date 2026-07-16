(() => {
  function wrapStatusLoader(name) {
    const original = window[name];
    if (typeof original !== "function") return;
    let pending = false;
    window[name] = async (...args) => {
      if (pending) return null;
      pending = true;
      try {
        return await original(...args);
      } finally {
        pending = false;
      }
    };
  }

  wrapStatusLoader("loadTrainingStatus");
  wrapStatusLoader("loadZeroStatus");

  function improveNetworkMessage() {
    const trainingBadge = document.querySelector("#training-status-badge");
    const trainingMessage = document.querySelector("#training-message");
    if (
      trainingBadge?.textContent === "unavailable" &&
      trainingMessage?.textContent === "Failed to fetch"
    ) {
      trainingBadge.textContent = "reconnecting";
      trainingMessage.textContent =
        "The browser temporarily cannot reach the local server. The training process may still be running; automatic polling will retry.";
    }

    const ready = document.querySelector("#zero-ready");
    const note = document.querySelector("#zero-status-note");
    if (ready?.textContent === "access required" && note?.textContent === "Failed to fetch") {
      ready.textContent = "reconnecting";
      note.textContent =
        "The local server is temporarily unreachable. This is a connection or CPU-load issue, not an access-token rejection.";
    }
  }

  window.setInterval(improveNetworkMessage, 400);
})();

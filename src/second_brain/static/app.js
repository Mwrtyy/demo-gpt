const $ = (selector) => document.querySelector(selector);

const state = {
  accessToken: localStorage.getItem("secondBrainAccessToken") || "",
  adminToken: localStorage.getItem("secondBrainAdminToken") || "",
};

$("#access-token").value = state.accessToken;
$("#admin-token").value = state.adminToken;

function headers(admin = false) {
  const result = { "Content-Type": "application/json" };
  if (state.accessToken) result["X-Access-Token"] = state.accessToken;
  if (admin && state.adminToken) result["X-Admin-Token"] = state.adminToken;
  return result;
}

async function api(path, options = {}, admin = false) {
  const response = await fetch(path, {
    ...options,
    headers: { ...headers(admin), ...(options.headers || {}) },
  });
  let payload = null;
  try {
    payload = await response.json();
  } catch {
    payload = { detail: await response.text() };
  }
  if (!response.ok) throw new Error(payload.detail || `Erreur HTTP ${response.status}`);
  return payload;
}

function toast(message, error = false) {
  const node = $("#toast");
  node.textContent = message;
  node.classList.toggle("error", error);
  node.classList.add("visible");
  clearTimeout(window.__toastTimer);
  window.__toastTimer = setTimeout(() => node.classList.remove("visible"), 3500);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function scrollMessages() {
  const box = $("#messages");
  box.scrollTop = box.scrollHeight;
}

function appendMessage(role, text, metadata = {}) {
  const article = document.createElement("article");
  article.className = `message ${role}`;
  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "VOUS" : "SB";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = `<p>${escapeHtml(text)}</p>`;

  if (metadata.interactionId) {
    const meta = document.createElement("div");
    meta.className = "message-meta";
    meta.textContent = `Interaction #${metadata.interactionId} · prompt v${metadata.promptVersion} · ${metadata.memoriesUsed} souvenir(s)`;
    bubble.appendChild(meta);

    const actions = document.createElement("div");
    actions.className = "feedback-actions";
    actions.innerHTML = `
      <button type="button" data-score="1">Utile</button>
      <button type="button" data-score="0.5">Moyen</button>
      <button type="button" data-score="0">À corriger</button>
    `;
    actions.querySelectorAll("button").forEach((button) => {
      button.addEventListener("click", async () => {
        try {
          await api("/api/feedback", {
            method: "POST",
            body: JSON.stringify({
              interaction_id: metadata.interactionId,
              score: Number(button.dataset.score),
              note: "",
            }),
          });
          actions.innerHTML = "<span class='message-meta'>Feedback enregistré.</span>";
          toast("Feedback enregistré.");
          loadStatus();
        } catch (error) {
          toast(error.message, true);
        }
      });
    });
    bubble.appendChild(actions);
  }

  article.append(avatar, bubble);
  $("#messages").appendChild(article);
  scrollMessages();
  return article;
}

function setLoading(button, loading, label = "Chargement…") {
  if (loading) {
    button.dataset.originalLabel = button.textContent;
    button.textContent = label;
    button.disabled = true;
  } else {
    button.textContent = button.dataset.originalLabel || button.textContent;
    button.disabled = false;
  }
}

async function loadStatus() {
  try {
    const data = await api("/api/status");
    $("#model-value").textContent = data.model;
    $("#version-value").textContent = `v${data.prompt_version}`;
    $("#memory-value").textContent = `${data.memory.interactions} échanges · ${data.memory.facts} faits`;
    $("#feedback-value").textContent = data.memory.average_feedback == null
      ? "aucune note"
      : `${Math.round(data.memory.average_feedback * 100)} %`;
  } catch (error) {
    $("#model-value").textContent = "accès requis";
    toast(error.message, true);
  }
}

$("#chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = $("#message-input");
  const message = input.value.trim();
  if (!message) return;

  appendMessage("user", message);
  input.value = "";
  input.style.height = "auto";

  const button = $("#send-button");
  setLoading(button, true, "Réflexion…");
  const placeholder = appendMessage("assistant", "…");

  try {
    const data = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({ message }),
    });
    placeholder.remove();
    appendMessage("assistant", data.answer, {
      interactionId: data.interaction_id,
      promptVersion: data.prompt_version,
      memoriesUsed: data.memories_used,
    });
    loadStatus();
  } catch (error) {
    placeholder.querySelector("p").textContent = `Erreur : ${error.message}`;
    toast(error.message, true);
  } finally {
    setLoading(button, false);
    input.focus();
  }
});

$("#message-input").addEventListener("input", (event) => {
  event.target.style.height = "auto";
  event.target.style.height = `${Math.min(event.target.scrollHeight, 180)}px`;
});

$("#message-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("#chat-form").requestSubmit();
  }
});

$("#save-fact").addEventListener("click", async () => {
  const button = $("#save-fact");
  const content = $("#fact-input").value.trim();
  const importance = Number($("#fact-importance").value);
  if (!content) return toast("Écris d’abord un souvenir.", true);

  setLoading(button, true);
  try {
    const data = await api("/api/facts", {
      method: "POST",
      body: JSON.stringify({ content, importance }),
    });
    $("#fact-input").value = "";
    toast(`Souvenir #${data.fact_id} enregistré.`);
    loadStatus();
  } catch (error) {
    toast(error.message, true);
  } finally {
    setLoading(button, false);
  }
});

$("#save-tokens").addEventListener("click", () => {
  state.accessToken = $("#access-token").value.trim();
  state.adminToken = $("#admin-token").value.trim();
  localStorage.setItem("secondBrainAccessToken", state.accessToken);
  localStorage.setItem("secondBrainAdminToken", state.adminToken);
  toast("Jetons enregistrés dans ce navigateur.");
  loadStatus();
});

$("#refresh-status").addEventListener("click", loadStatus);

$("#history-button").addEventListener("click", async () => {
  const dialog = $("#history-dialog");
  const list = $("#history-list");
  list.innerHTML = "<p>Chargement…</p>";
  dialog.showModal();
  try {
    const data = await api("/api/history?limit=40");
    if (!data.items.length) {
      list.innerHTML = "<p>Aucune interaction enregistrée.</p>";
      return;
    }
    list.innerHTML = data.items.map((item) => `
      <article class="history-item">
        <strong>#${item.id} · ${escapeHtml(item.user_input)}</strong>
        <p>${escapeHtml(item.answer)}</p>
        <div class="message-meta">prompt v${item.prompt_version} · ${escapeHtml(item.created_at)}</div>
      </article>
    `).join("");
  } catch (error) {
    list.innerHTML = `<p>${escapeHtml(error.message)}</p>`;
  }
});

$("#improve-button").addEventListener("click", () => {
  $("#improve-result").hidden = true;
  $("#improve-dialog").showModal();
});

$("#confirm-improve").addEventListener("click", async () => {
  const button = $("#confirm-improve");
  const result = $("#improve-result");
  setLoading(button, true, "Évaluation en cours…");
  result.hidden = false;
  result.textContent = "Le cycle peut effectuer plusieurs appels au modèle. Ne ferme pas cette page.";
  try {
    const data = await api(
      "/api/improve",
      {
        method: "POST",
        body: JSON.stringify({
          auto_promote: $("#auto-promote").checked,
          minimum_gain: 0.02,
          maximum_case_regression: 0.10,
        }),
      },
      true,
    );
    result.textContent = JSON.stringify(data, null, 2);
    toast(data.promoted ? "Nouvelle version promue." : "Candidate évaluée.");
    loadStatus();
  } catch (error) {
    result.textContent = `Erreur : ${error.message}`;
    toast(error.message, true);
  } finally {
    setLoading(button, false);
  }
});

document.querySelectorAll("[data-close]").forEach((button) => {
  button.addEventListener("click", () => {
    document.getElementById(button.dataset.close).close();
  });
});

loadStatus();

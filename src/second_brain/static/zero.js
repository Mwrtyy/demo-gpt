const $ = (selector) => document.querySelector(selector);

const zeroState = {
  accessToken: localStorage.getItem("secondBrainAccessToken") || "",
  adminToken: localStorage.getItem("secondBrainAdminToken") || "",
  generations: [],
  lastActivatedCheckpoint: null,
};

$("#zero-access-token").value = zeroState.accessToken;
$("#zero-admin-token").value = zeroState.adminToken;

function zeroHeaders(admin = false, json = true) {
  const result = {};
  if (json) result["Content-Type"] = "application/json";
  if (zeroState.accessToken) result["X-Access-Token"] = zeroState.accessToken;
  if (admin && zeroState.adminToken) result["X-Admin-Token"] = zeroState.adminToken;
  return result;
}

async function zeroApi(path, options = {}, admin = false, json = true) {
  const response = await fetch(path, {
    ...options,
    headers: { ...zeroHeaders(admin, json), ...(options.headers || {}) },
  });
  let payload;
  try {
    payload = await response.json();
  } catch {
    payload = { detail: await response.text() };
  }
  if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
  return payload;
}

function zeroToast(message, error = false) {
  const node = $("#zero-toast");
  node.textContent = message;
  node.classList.toggle("error", error);
  node.classList.add("visible");
  clearTimeout(window.__zeroToastTimer);
  window.__zeroToastTimer = setTimeout(() => node.classList.remove("visible"), 4000);
}

function setBusy(button, busy, label) {
  if (busy) {
    button.dataset.original = button.textContent;
    button.textContent = label;
    button.disabled = true;
  } else {
    button.textContent = button.dataset.original || button.textContent;
    button.disabled = false;
  }
}

function formatNumber(value) {
  return value == null ? "—" : new Intl.NumberFormat("en-US").format(value);
}

function formatBytes(value) {
  if (value == null) return "—";
  const units = ["B", "KB", "MB", "GB"];
  let amount = value;
  let unit = 0;
  while (amount >= 1024 && unit < units.length - 1) {
    amount /= 1024;
    unit += 1;
  }
  return `${amount.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function formatDuration(seconds) {
  if (seconds == null || !Number.isFinite(Number(seconds))) return "—";
  let remaining = Math.max(0, Math.round(Number(seconds)));
  const days = Math.floor(remaining / 86400);
  remaining %= 86400;
  const hours = Math.floor(remaining / 3600);
  remaining %= 3600;
  const minutes = Math.floor(remaining / 60);
  const parts = [];
  if (days) parts.push(`${days}d`);
  if (hours || days) parts.push(`${hours}h`);
  parts.push(`${minutes}m`);
  return parts.join(" ");
}

function renderStatus(data) {
  $("#zero-ready").textContent = data.ready ? "yes" : "no";
  $("#zero-ready").style.color = data.ready ? "var(--accent-2)" : "#ff9aa8";
  $("#zero-device").textContent = data.device || "—";
  $("#zero-parameters").textContent = formatNumber(data.parameters);
  $("#zero-step").textContent = data.step == null ? "—" : formatNumber(data.step);
  $("#zero-validation").textContent = data.best_validation == null
    ? "—"
    : Number(data.best_validation).toFixed(4);
  $("#zero-path").textContent = data.checkpoint_present
    ? `${data.checkpoint_path} · ${formatBytes(data.checkpoint_size_bytes)}`
    : data.checkpoint_path;

  let note = "Checkpoint loaded and ready for local generation.";
  if (!data.dependencies_available) note = data.dependency_error;
  else if (!data.checkpoint_present) note = "No checkpoint found. Train one or upload latest.pt below.";
  else if (!data.ready) note = data.load_error || "Checkpoint exists but could not be loaded.";
  $("#zero-status-note").textContent = note;
}

async function loadZeroStatus(showError = true) {
  try {
    const data = await zeroApi("/api/zero/status");
    renderStatus(data);
    return data;
  } catch (error) {
    $("#zero-ready").textContent = "access required";
    $("#zero-status-note").textContent = error.message;
    if (showError) zeroToast(error.message, true);
    return null;
  }
}

function selectedGeneration() {
  return zeroState.generations.find((item) => item.id === $("#training-generation").value);
}

function applyGenerationDefaults() {
  const generation = selectedGeneration();
  if (!generation) return;
  $("#training-max-steps").value = generation.default_max_steps;
  $("#training-target-validation").value = generation.target_validation;
}

async function loadTrainingCatalog() {
  try {
    const data = await zeroApi("/api/zero/training/catalog");
    zeroState.generations = data.generations || [];
    const select = $("#training-generation");
    const previous = select.value || "level1";
    select.innerHTML = zeroState.generations.map((generation) => (
      `<option value="${generation.id}">${generation.name}</option>`
    )).join("");
    select.value = zeroState.generations.some((item) => item.id === previous)
      ? previous
      : (zeroState.generations[0]?.id || "");
    applyGenerationDefaults();
  } catch (error) {
    $("#training-message").textContent = error.message;
  }
}

function renderTrainingStatus(data) {
  const status = data.status || "idle";
  const active = Boolean(data.active);
  const percent = Number(data.progress_percent || 0);
  $("#training-status-badge").textContent = status.replaceAll("_", " ");
  $("#training-status-badge").classList.toggle("training-failed", status === "failed");
  $("#training-stage").textContent = (data.stage || status).replaceAll("_", " ");
  $("#training-progress-text").textContent = `${percent.toFixed(percent < 1 ? 2 : 1)}%`;
  $("#training-progress-bar").style.width = `${Math.min(100, Math.max(0, percent))}%`;

  $("#training-generation-value").textContent = data.generation_name || data.generation || "—";
  $("#training-step").textContent = data.step == null
    ? "—"
    : `${formatNumber(data.step)} / ${formatNumber(data.max_steps)}`;
  $("#training-loss").textContent = data.loss == null ? "—" : Number(data.loss).toFixed(4);
  $("#training-validation").textContent = data.validation_loss == null
    ? "—"
    : Number(data.validation_loss).toFixed(4);
  $("#training-best").textContent = data.best_validation == null
    ? "—"
    : Number(data.best_validation).toFixed(4);
  $("#training-speed").textContent = data.sequences_per_second == null
    ? "—"
    : `${Number(data.sequences_per_second).toFixed(2)} seq/s`;
  $("#training-eta").textContent = formatDuration(data.eta_seconds);
  $("#training-process").textContent = data.pid
    ? `PID ${data.pid}${data.process_alive ? " · alive" : ""}`
    : "—";
  $("#training-log").textContent = data.log_tail || "Waiting for training output…";
  $("#training-log").scrollTop = $("#training-log").scrollHeight;
  $("#training-message").textContent = data.error || data.growth_message || "";

  $("#training-start").disabled = active;
  $("#training-pause").disabled = !["running", "resume_requested"].includes(status);
  $("#training-resume").disabled = !["paused", "pause_requested"].includes(status);
  $("#training-stop").disabled = !active;

  if (data.activated_checkpoint && data.activated_checkpoint !== zeroState.lastActivatedCheckpoint) {
    zeroState.lastActivatedCheckpoint = data.activated_checkpoint;
    loadZeroStatus(false);
  }
}

async function loadTrainingStatus(showError = false) {
  try {
    const data = await zeroApi("/api/zero/training/status");
    renderTrainingStatus(data);
    return data;
  } catch (error) {
    $("#training-status-badge").textContent = "unavailable";
    $("#training-message").textContent = error.message;
    if (showError) zeroToast(error.message, true);
    return null;
  }
}

async function trainingControl(action, button, busyLabel) {
  if (!zeroState.adminToken) return zeroToast("An administrator token is required.", true);
  setBusy(button, true, busyLabel);
  try {
    const data = await zeroApi(
      `/api/zero/training/${action}`,
      { method: "POST", body: "{}" },
      true,
    );
    renderTrainingStatus(data);
    zeroToast(`Training command accepted: ${action}.`);
  } catch (error) {
    zeroToast(error.message, true);
  } finally {
    setBusy(button, false);
  }
}

$("#zero-save-tokens").addEventListener("click", () => {
  zeroState.accessToken = $("#zero-access-token").value.trim();
  zeroState.adminToken = $("#zero-admin-token").value.trim();
  localStorage.setItem("secondBrainAccessToken", zeroState.accessToken);
  localStorage.setItem("secondBrainAdminToken", zeroState.adminToken);
  zeroToast("Tokens saved in this browser.");
  loadZeroStatus();
  loadTrainingStatus();
});

$("#zero-refresh").addEventListener("click", () => {
  loadZeroStatus();
  loadTrainingStatus();
});

$("#training-generation").addEventListener("change", applyGenerationDefaults);

$("#training-start").addEventListener("click", async () => {
  const button = $("#training-start");
  if (!zeroState.adminToken) return zeroToast("An administrator token is required.", true);
  setBusy(button, true, "Starting…");
  try {
    const data = await zeroApi(
      "/api/zero/training/start",
      {
        method: "POST",
        body: JSON.stringify({
          generation: $("#training-generation").value,
          max_steps: Number($("#training-max-steps").value),
          target_validation: Number($("#training-target-validation").value),
          max_parameters: Number($("#training-max-parameters").value),
          auto_prepare: $("#training-auto-prepare").checked,
          resume_existing: $("#training-resume-existing").checked,
          auto_activate_best: $("#training-auto-activate").checked,
          auto_advance: $("#training-auto-advance").checked,
          initialize_from_previous: $("#training-init-previous").checked,
          resume_after_restart: $("#training-resume-restart").checked,
        }),
      },
      true,
    );
    renderTrainingStatus(data);
    zeroToast("Autonomous training started.");
  } catch (error) {
    zeroToast(error.message, true);
  } finally {
    setBusy(button, false);
  }
});

$("#training-pause").addEventListener("click", () => (
  trainingControl("pause", $("#training-pause"), "Pause requested…")
));
$("#training-resume").addEventListener("click", () => (
  trainingControl("resume", $("#training-resume"), "Resuming…")
));
$("#training-stop").addEventListener("click", () => (
  trainingControl("stop", $("#training-stop"), "Stop requested…")
));

$("#zero-upload").addEventListener("click", async () => {
  const button = $("#zero-upload");
  const file = $("#zero-checkpoint").files[0];
  if (!file) return zeroToast("Choose a .pt checkpoint first.", true);
  if (!zeroState.adminToken) return zeroToast("An administrator token is required.", true);

  const form = new FormData();
  form.append("checkpoint", file);
  setBusy(button, true, "Uploading and validating…");
  try {
    const data = await zeroApi(
      "/api/zero/checkpoint",
      { method: "POST", body: form },
      true,
      false,
    );
    renderStatus(data.status);
    $("#zero-checkpoint").value = "";
    zeroToast("Checkpoint activated successfully.");
  } catch (error) {
    zeroToast(error.message, true);
  } finally {
    setBusy(button, false);
  }
});

$("#zero-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = $("#zero-generate");
  const output = $("#zero-output");
  const metadata = $("#zero-metadata");
  setBusy(button, true, "Generating locally…");
  output.textContent = "Running our checkpoint…";
  metadata.textContent = "";
  $("#zero-speed").textContent = "working";

  try {
    const data = await zeroApi("/api/zero/generate", {
      method: "POST",
      body: JSON.stringify({
        prompt: $("#zero-prompt").value,
        max_new_tokens: Number($("#zero-max-tokens").value),
        temperature: Number($("#zero-temperature").value),
        top_k: Number($("#zero-top-k").value),
        seed: Number($("#zero-seed").value),
      }),
    });
    output.textContent = data.continuation || "[The model produced no decodable continuation.]";
    metadata.textContent = `${data.new_tokens} new bytes · ${data.elapsed_seconds}s · ${data.device} · step ${data.step ?? "unknown"}`;
    $("#zero-speed").textContent = `${data.tokens_per_second} bytes/s`;
  } catch (error) {
    output.textContent = `Error: ${error.message}`;
    $("#zero-speed").textContent = "failed";
    zeroToast(error.message, true);
  } finally {
    setBusy(button, false);
  }
});

$("#zero-copy").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText($("#zero-output").textContent);
    zeroToast("Output copied.");
  } catch {
    zeroToast("Clipboard access was refused.", true);
  }
});

Promise.all([loadTrainingCatalog(), loadTrainingStatus(false), loadZeroStatus(false)]);
window.setInterval(() => loadTrainingStatus(false), 2500);
window.setInterval(() => loadZeroStatus(false), 10000);

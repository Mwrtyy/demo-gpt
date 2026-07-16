const $ = (selector) => document.querySelector(selector);

const zeroState = {
  accessToken: localStorage.getItem("secondBrainAccessToken") || "",
  adminToken: localStorage.getItem("secondBrainAdminToken") || "",
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

$("#zero-save-tokens").addEventListener("click", () => {
  zeroState.accessToken = $("#zero-access-token").value.trim();
  zeroState.adminToken = $("#zero-admin-token").value.trim();
  localStorage.setItem("secondBrainAccessToken", zeroState.accessToken);
  localStorage.setItem("secondBrainAdminToken", zeroState.adminToken);
  zeroToast("Tokens saved in this browser.");
  loadZeroStatus();
});

$("#zero-refresh").addEventListener("click", () => loadZeroStatus());

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

loadZeroStatus(false);

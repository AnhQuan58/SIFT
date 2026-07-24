const fpsEl = document.getElementById("stat-fps");
const kpEl = document.getElementById("stat-kp");
const backendEl = document.getElementById("stat-backend");
const pillEl = document.getElementById("backend-status");
const selectEl = document.getElementById("backend-select");

async function pollStatus() {
  try {
    const res = await fetch("/status");
    if (!res.ok) return;
    const data = await res.json();
    fpsEl.textContent = data.fps.toFixed(1);
    kpEl.textContent = data.n_keypoints;
    backendEl.textContent = data.backend;

    // Once the camera loop has actually switched, drop the "busy" state.
    if (data.backend === selectEl.value) {
      pillEl.textContent = "ready";
      pillEl.classList.remove("busy");
    }
  } catch (err) {
    // Camera loop / server may not be up yet; stay quiet and retry.
  }
}

selectEl.addEventListener("change", async () => {
  const backend = selectEl.value;
  pillEl.textContent = "switching…";
  pillEl.classList.add("busy");
  try {
    await fetch("/set_backend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ backend }),
    });
  } catch (err) {
    pillEl.textContent = "error";
  }
});

setInterval(pollStatus, 500);
pollStatus();

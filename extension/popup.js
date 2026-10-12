document.addEventListener("DOMContentLoaded", () => {
  const btnOpenDash = document.getElementById("btn-open-dash");
  const btnOpenMeet = document.getElementById("btn-open-meet");
  const selSensitivity = document.getElementById("sel-sensitivity");
  const selAutoHud = document.getElementById("sel-autohud");

  // Load saved options
  if (chrome.storage && chrome.storage.local) {
    chrome.storage.local.get(["sp_sensitivity", "sp_autohud"], (res) => {
      if (res.sp_sensitivity) selSensitivity.value = res.sp_sensitivity;
      if (res.sp_autohud) selAutoHud.value = res.sp_autohud;
    });
  }

  selSensitivity.addEventListener("change", () => {
    if (chrome.storage && chrome.storage.local) {
      chrome.storage.local.set({ sp_sensitivity: selSensitivity.value });
    }
  });

  selAutoHud.addEventListener("change", () => {
    if (chrome.storage && chrome.storage.local) {
      chrome.storage.local.set({ sp_autohud: selAutoHud.value });
    }
  });

  btnOpenDash.addEventListener("click", () => {
    chrome.tabs.create({ url: "http://localhost:8000" });
  });

  btnOpenMeet.addEventListener("click", () => {
    chrome.tabs.create({ url: "https://meet.google.com" });
  });
});

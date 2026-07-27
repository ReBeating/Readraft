(() => {
  "use strict";

  const dialog = document.querySelector("[data-model-settings-dialog]");
  if (!dialog) return;

  const frame = dialog.querySelector("[data-model-settings-frame]");
  const loading = dialog.querySelector("[data-model-settings-loading]");
  const closeButton = dialog.querySelector("[data-model-settings-close]");
  let lastOpener = null;

  function openDialog(opener) {
    lastOpener = opener;
    if (!frame.getAttribute("src")) {
      frame.setAttribute("src", dialog.dataset.settingsUrl);
    }
    if (typeof dialog.showModal === "function") {
      dialog.showModal();
    } else {
      dialog.setAttribute("open", "");
    }
  }

  function closeDialog() {
    if (typeof dialog.close === "function") {
      dialog.close();
    } else {
      dialog.removeAttribute("open");
      lastOpener?.focus();
    }
  }

  async function refreshModelChoices() {
    const selects = [
      ...document.querySelectorAll("select[data-model-choice]"),
    ];
    if (!selects.length) return;

    try {
      const response = await fetch("/api/settings/chat-models", {
        credentials: "same-origin",
        cache: "no-store",
        headers: {"Accept": "application/json"},
      });
      if (!response.ok) return;
      const payload = await response.json();
      const groups = Array.isArray(payload.groups) ? payload.groups : [];

      selects.forEach((select) => {
        const previousValue = select.value;
        select.replaceChildren();
        groups.forEach((group) => {
          const optionGroup = document.createElement("optgroup");
          optionGroup.label = group.label;
          group.models.forEach((model) => {
            const option = document.createElement("option");
            option.value = model.value;
            option.textContent = model.id;
            optionGroup.append(option);
          });
          select.append(optionGroup);
        });

        const availableValues = new Set(
          [...select.options].map((option) => option.value),
        );
        if (availableValues.has(previousValue)) {
          select.value = previousValue;
        } else if (availableValues.has(payload.default)) {
          select.value = payload.default;
        }
        if (!select.options.length) {
          const option = document.createElement("option");
          option.value = "";
          option.textContent = "尚未配置模型";
          select.append(option);
        }
        select.disabled = groups.length === 0;
      });
    } catch (_) {
      // The saved settings remain valid; the current picker refreshes next load.
    }
  }

  document.addEventListener("click", (event) => {
    const opener = event.target.closest("[data-model-settings-open]");
    if (!opener) return;
    event.preventDefault();
    openDialog(opener);
  });

  closeButton.addEventListener("click", closeDialog);
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) closeDialog();
  });
  dialog.addEventListener("close", () => {
    lastOpener?.focus();
  });
  frame.addEventListener("load", () => {
    loading.hidden = true;
    frame.hidden = false;
  });
  window.addEventListener("message", (event) => {
    if (
      event.origin !== window.location.origin
      || event.source !== frame.contentWindow
      || event.data?.type !== "readraft:model-settings-updated"
    ) {
      return;
    }
    refreshModelChoices();
  });
})();

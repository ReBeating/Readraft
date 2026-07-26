(() => {
  "use strict";

  const THEME_KEY = "novel-workbench-theme";
  const root = document.documentElement;

  function storedTheme() {
    try {
      return localStorage.getItem(THEME_KEY) === "night" ? "night" : "day";
    } catch (_) {
      return "day";
    }
  }

  function applyTheme(theme) {
    const nextTheme = theme === "night" ? "night" : "day";
    root.dataset.theme = nextTheme;
    document.querySelectorAll("[data-studio-theme]").forEach((node) => {
      node.dataset.theme = nextTheme;
    });
    document.querySelectorAll("[data-theme-toggle]").forEach((button) => {
      const night = nextTheme === "night";
      button.setAttribute(
        "aria-label",
        night ? "切换到白天模式" : "切换到夜间模式",
      );
      button.querySelectorAll("[data-theme-icon]").forEach((icon) => {
        icon.toggleAttribute(
          "hidden",
          icon.getAttribute("data-theme-icon") !==
            (night ? "day" : "night"),
        );
      });
    });
    try {
      localStorage.setItem(THEME_KEY, nextTheme);
    } catch (_) {
      // Private browsing can disable localStorage; the current page still works.
    }
  }

  applyTheme(storedTheme());
  document.querySelectorAll("[data-theme-toggle]").forEach((button) => {
    button.addEventListener("click", () => {
      applyTheme(root.dataset.theme === "night" ? "day" : "night");
    });
  });

  let pendingDestructiveForm = null;

  function confirmationDialog() {
    let dialog = document.querySelector("[data-studio-confirm-dialog]");
    if (dialog) return dialog;

    dialog = document.createElement("dialog");
    dialog.className = "studio-confirm-dialog";
    dialog.dataset.studioConfirmDialog = "";
    dialog.setAttribute("aria-labelledby", "studio-confirm-title");
    dialog.setAttribute("aria-describedby", "studio-confirm-message");
    dialog.innerHTML = `
      <form method="dialog">
        <div class="studio-confirm-symbol" aria-hidden="true">
          <svg viewBox="0 0 24 24">
            <path d="M4 7h16M9 7V4h6v3M7 7l1 13h8l1-13"></path>
          </svg>
        </div>
        <div class="studio-confirm-copy">
          <h2 id="studio-confirm-title" data-confirm-title>确认删除</h2>
          <p id="studio-confirm-message" data-confirm-message></p>
        </div>
        <div class="studio-confirm-actions">
          <button class="studio-button" type="submit" value="cancel">取消</button>
          <button class="studio-button danger" type="submit" value="confirm" data-confirm-submit>确认删除</button>
        </div>
      </form>
    `;
    dialog.addEventListener("close", () => {
      const target = pendingDestructiveForm;
      pendingDestructiveForm = null;
      if (dialog.returnValue !== "confirm" || !target) return;
      target.dataset.confirmed = "true";
      target.requestSubmit();
    });
    document.body.append(dialog);
    return dialog;
  }

  function requestDestructiveConfirmation(form, options) {
    const dialog = confirmationDialog();
    dialog.querySelector("[data-confirm-title]").textContent = options.title;
    dialog.querySelector("[data-confirm-message]").textContent =
      options.message;
    dialog.querySelector("[data-confirm-submit]").textContent =
      options.confirmLabel;
    pendingDestructiveForm = form;
    dialog.returnValue = "";
    if (typeof dialog.showModal === "function") {
      dialog.showModal();
    } else {
      dialog.setAttribute("open", "");
    }
  }

  function bindDestructiveConfirmation(selector, options) {
    document.querySelectorAll(selector).forEach((form) => {
      form.addEventListener("submit", (event) => {
        if (form.dataset.confirmed === "true") {
          delete form.dataset.confirmed;
          return;
        }
        event.preventDefault();
        requestDestructiveConfirmation(form, options);
      });
    });
  }

  bindDestructiveConfirmation("[data-delete-project-form]", {
    title: "删除这部作品？",
    message:
      "作品设定、章节正文、全部版本及对应对话都会被永久删除，且无法恢复。",
    confirmLabel: "删除作品",
  });

  bindDestructiveConfirmation("[data-delete-conversation-form]", {
    title: "删除这段对话？",
    message:
      "这段对话及其中的消息会被永久删除；已经提交到正文的版本仍会保留。",
    confirmLabel: "删除对话",
  });

  bindDestructiveConfirmation("[data-delete-provider-form]", {
    title: "删除这个模型配置？",
    message: "保存的凭据、接口地址和模型列表都会被永久删除。",
    confirmLabel: "删除配置",
  });

  const workbench = document.querySelector("[data-workbench]");
  if (!workbench) return;

  const mobileQuery = window.matchMedia("(max-width: 760px)");
  const mobileAIStateKey =
    `novel-workbench-ai-open:${workbench.dataset.projectId || "current"}`;
  let shouldOpenOnboardingAI =
    workbench.dataset.onboarding === "true";
  const panelState = {
    directory: !mobileQuery.matches,
    ai: !mobileQuery.matches,
  };

  function rememberedMobileAIState() {
    try {
      return sessionStorage.getItem(mobileAIStateKey) === "true";
    } catch (_) {
      return false;
    }
  }

  function rememberMobileAIState(isOpen) {
    try {
      sessionStorage.setItem(mobileAIStateKey, isOpen ? "true" : "false");
    } catch (_) {
      // The current page remains usable if session storage is unavailable.
    }
  }

  function setPanel(name, isOpen, options = {}) {
    const panel = workbench.querySelector(`[data-panel="${name}"]`);
    const button = workbench.querySelector(`[data-panel-toggle="${name}"]`);
    if (!panel || !button) return;

    if (
      mobileQuery.matches &&
      isOpen &&
      !options.skipPeerClose
    ) {
      const peer = name === "directory" ? "ai" : "directory";
      setPanel(peer, false, { skipPeerClose: true });
    }

    panelState[name] = Boolean(isOpen);
    panel.dataset.open = isOpen ? "true" : "false";
    button.setAttribute("aria-pressed", isOpen ? "true" : "false");
    button.setAttribute(
      "aria-label",
      `${isOpen ? "关闭" : "打开"}${name === "directory" ? "目录" : " AI 共创"}`,
    );
    workbench.classList.toggle(`${name}-closed`, !isOpen);
    if (
      name === "ai" &&
      mobileQuery.matches &&
      options.persist !== false
    ) {
      rememberMobileAIState(Boolean(isOpen));
    }
  }

  function resetPanelsForViewport() {
    if (mobileQuery.matches) {
      const shouldOpenAI =
        shouldOpenOnboardingAI || rememberedMobileAIState();
      setPanel("directory", false, {
        skipPeerClose: true,
        persist: false,
      });
      setPanel("ai", shouldOpenAI, {
        skipPeerClose: true,
      });
      shouldOpenOnboardingAI = false;
    } else {
      setPanel("directory", true, { skipPeerClose: true });
      setPanel("ai", true, { skipPeerClose: true });
    }
  }

  resetPanelsForViewport();
  mobileQuery.addEventListener?.("change", resetPanelsForViewport);

  workbench.querySelectorAll("[data-panel-toggle]").forEach((button) => {
    button.addEventListener("click", () => {
      const name = button.getAttribute("data-panel-toggle");
      setPanel(name, !panelState[name]);
    });
  });

  const settingsTabs = Array.from(
    workbench.querySelectorAll("[data-settings-tab]"),
  );
  const settingsPanels = Array.from(
    workbench.querySelectorAll("[data-settings-panel]"),
  );
  const settingsTabInput = workbench.querySelector(
    "[data-settings-tab-input]",
  );
  const returnSettingsTabInput = workbench.querySelector(
    'input[name="return_settings_tab"]',
  );

  function activateSettingsTab(key, options = {}) {
    if (!settingsTabs.some((button) => button.dataset.settingsTab === key)) {
      return;
    }
    settingsTabs.forEach((button) => {
      const active = button.dataset.settingsTab === key;
      button.setAttribute("aria-selected", active ? "true" : "false");
      button.tabIndex = active ? 0 : -1;
    });
    settingsPanels.forEach((panel) => {
      panel.hidden = panel.dataset.settingsPanel !== key;
    });
    if (settingsTabInput) settingsTabInput.value = key;
    if (returnSettingsTabInput) returnSettingsTabInput.value = key;
    if (options.focus) {
      settingsTabs
        .find((button) => button.dataset.settingsTab === key)
        ?.focus();
    }
    if (options.updateUrl !== false) {
      const url = new URL(window.location.href);
      url.searchParams.set("settings_tab", key);
      window.history.replaceState({}, "", url);
    }
  }

  settingsTabs.forEach((button, index) => {
    button.addEventListener("click", () => {
      activateSettingsTab(button.dataset.settingsTab);
    });
    button.addEventListener("keydown", (event) => {
      let nextIndex = null;
      if (event.key === "ArrowRight") {
        nextIndex = (index + 1) % settingsTabs.length;
      } else if (event.key === "ArrowLeft") {
        nextIndex =
          (index - 1 + settingsTabs.length) % settingsTabs.length;
      } else if (event.key === "Home") {
        nextIndex = 0;
      } else if (event.key === "End") {
        nextIndex = settingsTabs.length - 1;
      }
      if (nextIndex === null) return;
      event.preventDefault();
      activateSettingsTab(
        settingsTabs[nextIndex].dataset.settingsTab,
        { focus: true },
      );
    });
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && mobileQuery.matches) {
      setPanel("directory", false, { skipPeerClose: true });
      setPanel("ai", false, { skipPeerClose: true });
    }
  });

  const chatScroll = workbench.querySelector("[data-chat-scroll]");
  if (chatScroll) {
    chatScroll.scrollTop = chatScroll.scrollHeight;
  }

  const manuscript = workbench.querySelector("[data-manuscript]");
  const autosaveForm = workbench.querySelector("[data-autosave-form]");
  const saveStatus = workbench.querySelector("[data-save-status]");
  const chatForm = workbench.querySelector("[data-chat-form]");
  const charCount = workbench.querySelector("[data-char-count]");
  let autosaveTimer = 0;
  let savePromise = null;
  let lastSavedContent = manuscript ? manuscript.value : "";

  function updateCharCount() {
    if (!manuscript || !charCount) return;
    const count = manuscript.value.replace(/\s+/gu, "").length;
    const target = Number(charCount.dataset.target || 0);
    charCount.textContent = target
      ? `${count} / 约 ${target} 字`
      : `${count} 字`;
  }

  function setSaveStatus(message, state = "") {
    if (!saveStatus) return;
    saveStatus.textContent = message;
    saveStatus.dataset.state = state;
  }

  function refreshVersionReference(html) {
    if (!chatForm || !html) return;
    const parsed = new DOMParser().parseFromString(html, "text/html");
    const versionId = parsed.querySelector(
      '[data-chat-form] input[name="source_version_id"]',
    );
    const sourceHash = parsed.querySelector(
      '[data-chat-form] input[name="source_hash"]',
    );
    if (versionId && sourceHash) {
      chatForm.elements.source_version_id.value = versionId.value;
      chatForm.elements.source_hash.value = sourceHash.value;
    }
  }

  async function saveManuscript(force = false) {
    if (!autosaveForm || !manuscript) return true;
    if (manuscript.value === lastSavedContent) return true;
    if (savePromise) {
      const saved = await savePromise;
      if (!saved) return false;
      return manuscript.value === lastSavedContent
        ? true
        : saveManuscript(force);
    }

    window.clearTimeout(autosaveTimer);
    setSaveStatus("保存中…", "saving");
    const contentAtStart = manuscript.value;
    savePromise = (async () => {
      try {
        const response = await fetch(autosaveForm.action, {
          method: "POST",
          body: new FormData(autosaveForm),
          credentials: "same-origin",
          headers: { "X-Requested-With": "XMLHttpRequest" },
        });
        if (!response.ok) throw new Error("save failed");
        const html = await response.text();
        refreshVersionReference(html);
        lastSavedContent = contentAtStart;
        if (manuscript.value === contentAtStart) {
          setSaveStatus("已保存", "saved");
        } else {
          setSaveStatus("有未保存修改", "dirty");
          autosaveTimer = window.setTimeout(() => saveManuscript(), 900);
        }
        return true;
      } catch (_) {
        setSaveStatus("保存失败，点击重试", "error");
        return false;
      } finally {
        savePromise = null;
      }
    })();
    return savePromise;
  }

  if (autosaveForm && manuscript) {
    manuscript.addEventListener("input", () => {
      updateCharCount();
      setSaveStatus("有未保存修改", "dirty");
      window.clearTimeout(autosaveTimer);
      autosaveTimer = window.setTimeout(() => saveManuscript(), 1200);
    });
    updateCharCount();

    saveStatus?.addEventListener("click", () => {
      if (saveStatus.dataset.state === "error") saveManuscript(true);
    });

    document.addEventListener("keydown", (event) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "s") {
        event.preventDefault();
        saveManuscript(true);
      }
    });

    window.addEventListener("beforeunload", (event) => {
      if (manuscript.value === lastSavedContent) return;
      event.preventDefault();
      event.returnValue = "";
    });
  }

  function manuscriptNeedsSave() {
    return Boolean(
      manuscript &&
        autosaveForm &&
        (savePromise || manuscript.value !== lastSavedContent),
    );
  }

  workbench
    .querySelectorAll("a[data-save-before-navigation]")
    .forEach((anchor) => {
      anchor.addEventListener("click", async (event) => {
        if (
          event.defaultPrevented ||
          event.button !== 0 ||
          event.metaKey ||
          event.ctrlKey ||
          event.shiftKey ||
          event.altKey ||
          !manuscriptNeedsSave()
        ) {
          return;
        }
        event.preventDefault();
        if (anchor.dataset.savePending === "true") return;
        anchor.dataset.savePending = "true";
        anchor.setAttribute("aria-disabled", "true");
        const saved = await saveManuscript(true);
        if (saved) {
          window.location.assign(anchor.href);
          return;
        }
        delete anchor.dataset.savePending;
        anchor.removeAttribute("aria-disabled");
      });
    });

  workbench
    .querySelectorAll("form[data-save-before-submit]")
    .forEach((form) => {
      form.addEventListener("submit", async (event) => {
        if (!manuscriptNeedsSave()) return;
        event.preventDefault();
        if (form.dataset.savePending === "true") return;
        form.dataset.savePending = "true";
        const submitter = event.submitter;
        if (submitter) submitter.disabled = true;
        const saved = await saveManuscript(true);
        if (saved) {
          if (submitter?.name) {
            const submitterValue = document.createElement("input");
            submitterValue.type = "hidden";
            submitterValue.name = submitter.name;
            submitterValue.value = submitter.value;
            form.append(submitterValue);
          }
          HTMLFormElement.prototype.submit.call(form);
          return;
        }
        delete form.dataset.savePending;
        if (submitter) submitter.disabled = false;
      });
    });

  if (chatForm) {
    const chatInput = chatForm.querySelector('textarea[name="question"]');

    chatInput?.addEventListener("keydown", (event) => {
      if (
        event.key !== "Enter" ||
        event.shiftKey ||
        event.isComposing ||
        event.keyCode === 229
      ) {
        return;
      }
      event.preventDefault();
      const submit = chatForm.querySelector('[type="submit"]');
      if (!submit || submit.disabled || !chatInput.value.trim()) return;
      if (typeof chatForm.requestSubmit === "function") {
        chatForm.requestSubmit(submit);
      } else {
        submit.click();
      }
    });
  }

  const quotePreview = workbench.querySelector("[data-quote-preview]");
  const quoteTextField = workbench.querySelector("[data-quote-text]");
  const quoteStartField = workbench.querySelector("[data-quote-start]");
  const quoteEndField = workbench.querySelector("[data-quote-end]");

  function clearQuote() {
    if (quoteTextField) quoteTextField.value = "";
    if (quoteStartField) quoteStartField.value = "";
    if (quoteEndField) quoteEndField.value = "";
    if (quotePreview) {
      quotePreview.hidden = true;
      const quote = quotePreview.querySelector("q");
      if (quote) quote.textContent = "";
    }
  }

  function captureSelection() {
    if (!manuscript || !quotePreview) return;
    const start = manuscript.selectionStart;
    const end = manuscript.selectionEnd;
    const selected = manuscript.value.slice(start, end).trim();
    if (!selected || selected.length > 2400) {
      if (!selected) clearQuote();
      return;
    }
    const codePointStart = Array.from(
      manuscript.value.slice(0, start),
    ).length;
    const codePointEnd = codePointStart + Array.from(
      manuscript.value.slice(start, end),
    ).length;
    quoteStartField.value = String(codePointStart);
    quoteEndField.value = String(codePointEnd);
    quoteTextField.value = manuscript.value.slice(start, end);
    quotePreview.querySelector("q").textContent =
      selected.length > 100 ? `${selected.slice(0, 100)}…` : selected;
    quotePreview.hidden = false;
  }

  if (manuscript && quotePreview) {
    manuscript.addEventListener("mouseup", captureSelection);
    manuscript.addEventListener("keyup", (event) => {
      if (event.shiftKey) captureSelection();
    });
    workbench
      .querySelector("[data-clear-quote]")
      ?.addEventListener("click", clearQuote);
  }

  if (chatForm) {
    chatForm.addEventListener("submit", async (event) => {
      if (
        manuscript &&
        autosaveForm &&
        manuscript.value !== lastSavedContent
      ) {
        event.preventDefault();
        const submit = chatForm.querySelector('[type="submit"]');
        if (submit) submit.disabled = true;
        const saved = await saveManuscript(true);
        if (saved) {
          HTMLFormElement.prototype.submit.call(chatForm);
        } else if (submit) {
          submit.disabled = false;
        }
      }
    });
  }

  const edgeCue = workbench.querySelector("[data-edge-cue]");
  let edgeAmount = 0;
  let edgeDirection = "";
  let edgeResetTimer = 0;
  let touchStartY = null;
  let chapterNavigationStarted = false;

  function showEdgeCue(direction, amount) {
    if (!edgeCue) return;
    const hasTarget =
      direction === "next"
        ? manuscript.dataset.nextUrl
        : manuscript.dataset.previousUrl;
    edgeCue.hidden = false;
    if (!hasTarget) {
      edgeCue.textContent =
        direction === "next" ? "已经是最后一章" : "已经是第一章";
      return;
    }
    edgeCue.textContent =
      amount >= 100
        ? "松开切换章节"
        : direction === "next"
          ? "继续向上滑，进入下一章"
          : "继续向下滑，返回上一章";
  }

  function resetEdgeCue(delay = 500) {
    window.clearTimeout(edgeResetTimer);
    edgeResetTimer = window.setTimeout(() => {
      edgeAmount = 0;
      edgeDirection = "";
      if (edgeCue) edgeCue.hidden = true;
    }, delay);
  }

  function navigateChapter(direction) {
    if (chapterNavigationStarted || !manuscript) return;
    const target =
      direction === "next"
        ? manuscript.dataset.nextUrl
        : manuscript.dataset.previousUrl;
    if (!target) {
      resetEdgeCue(900);
      return;
    }
    chapterNavigationStarted = true;
    saveManuscript(true).then((saved) => {
      if (saved) {
        window.location.assign(target);
      } else {
        chapterNavigationStarted = false;
      }
    });
  }

  if (manuscript) {
    manuscript.addEventListener(
      "wheel",
      (event) => {
        const atTop = manuscript.scrollTop <= 1;
        const atBottom =
          manuscript.scrollTop + manuscript.clientHeight >=
          manuscript.scrollHeight - 1;
        const direction =
          event.deltaY > 0 && atBottom
            ? "next"
            : event.deltaY < 0 && atTop
              ? "previous"
              : "";
        if (!direction) {
          edgeAmount = 0;
          edgeDirection = "";
          if (edgeCue) edgeCue.hidden = true;
          return;
        }
        event.preventDefault();
        if (edgeDirection !== direction) edgeAmount = 0;
        edgeDirection = direction;
        edgeAmount += Math.min(36, Math.abs(event.deltaY));
        showEdgeCue(direction, edgeAmount);
        if (edgeAmount >= 150) navigateChapter(direction);
        resetEdgeCue(650);
      },
      { passive: false },
    );

    manuscript.addEventListener(
      "touchstart",
      (event) => {
        touchStartY = event.touches[0]?.clientY ?? null;
        edgeAmount = 0;
      },
      { passive: true },
    );

    manuscript.addEventListener(
      "touchmove",
      (event) => {
        if (touchStartY === null) return;
        const currentY = event.touches[0]?.clientY;
        if (currentY === undefined) return;
        const distance = currentY - touchStartY;
        const atTop = manuscript.scrollTop <= 1;
        const atBottom =
          manuscript.scrollTop + manuscript.clientHeight >=
          manuscript.scrollHeight - 1;
        const direction =
          distance < 0 && atBottom
            ? "next"
            : distance > 0 && atTop
              ? "previous"
              : "";
        if (!direction) return;
        edgeDirection = direction;
        edgeAmount = Math.min(130, Math.abs(distance));
        showEdgeCue(direction, edgeAmount);
        if (edgeAmount > 18) event.preventDefault();
      },
      { passive: false },
    );

    manuscript.addEventListener("touchend", () => {
      if (edgeDirection && edgeAmount >= 100) {
        navigateChapter(edgeDirection);
      } else {
        resetEdgeCue();
      }
      touchStartY = null;
    });
  }

  const pendingMessageId = workbench.dataset.pendingMessageId;
  if (pendingMessageId) {
    let attempts = 0;
    const poll = async () => {
      attempts += 1;
      try {
        const response = await fetch(
          `/api/assistant/messages/${encodeURIComponent(pendingMessageId)}`,
          {
            credentials: "same-origin",
            headers: { Accept: "application/json" },
          },
        );
        if (!response.ok) throw new Error("poll failed");
        const data = await response.json();
        if (data.terminal) {
          window.location.reload();
          return;
        }
      } catch (_) {
        // A later poll can recover from a transient local connection error.
      }
      if (attempts < 600) window.setTimeout(poll, 1000);
    };
    window.setTimeout(poll, 700);
  }
})();

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

  bindDestructiveConfirmation(
    "[data-delete-project-form], [data-delete-work-form]",
    {
    title: "删除这部作品？",
    message:
      "main、所有 Tag、作品资料、分析与对话都会被永久删除，且无法恢复。",
    confirmLabel: "删除作品",
    },
  );

  bindDestructiveConfirmation("[data-delete-conversation-form]", {
    title: "删除这段对话？",
    message:
      "这段对话及其中的消息会被永久删除；已经提交到正文的版本仍会保留。",
    confirmLabel: "删除对话",
  });

  bindDestructiveConfirmation("[data-delete-version-form]", {
    title: "删除这个 Tag？",
    message:
      "这个固定版本的正文快照、分析与阅读对话都会被永久删除；main 和其他 Tag 不受影响。",
    confirmLabel: "删除 Tag",
  });

  bindDestructiveConfirmation("[data-delete-provider-form]", {
    title: "删除这个模型配置？",
    message: "保存的凭据、接口地址和模型列表都会被永久删除。",
    confirmLabel: "删除配置",
  });

  bindDestructiveConfirmation("[data-delete-archive-entry]", {
    title: "移除这条档案记录？",
    message:
      "记录会从作品档案中删除；如果它是已采纳设定，对写作模型的约束也会同时移除。",
    confirmLabel: "确认移除",
  });

  const unifiedImport = document.querySelector("[data-unified-import]");
  if (unifiedImport) {
    const fileInput = unifiedImport.querySelector("[data-import-file]");
    const archiveNote = unifiedImport.querySelector(
      "[data-import-archive-note]",
    );
    const updateImportKind = () => {
      const filename = fileInput?.files?.[0]?.name?.toLowerCase() || "";
      const isArchive = filename.endsWith(".zip");
      if (archiveNote) {
        archiveNote.hidden = !isArchive;
      }
    };
    fileInput?.addEventListener("change", updateImportKind);
    updateImportKind();
  }

  const workbench = document.querySelector("[data-workbench]");
  if (!workbench) return;

  if (window.location.hash) {
    let anchorId = window.location.hash.slice(1);
    try {
      anchorId = decodeURIComponent(anchorId);
    } catch (_) {
      // Keep the literal fragment if it is not valid percent-encoding.
    }
    const memoryCard = document.getElementById(anchorId);
    if (memoryCard?.matches("details.studio-story-memory-card")) {
      memoryCard.open = true;
      window.requestAnimationFrame(() => {
        memoryCard.scrollIntoView({ block: "start" });
      });
    }
  }

  const mobileQuery = window.matchMedia("(max-width: 760px)");
  const mobileAIStateKey =
    `novel-workbench-ai-open:${
      workbench.dataset.projectId ||
      workbench.dataset.documentId ||
      "current"
    }`;
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
    const panelLabel =
      button.dataset.panelLabel ||
      (name === "directory" ? "目录" : "AI 共创");
    button.setAttribute(
      "aria-label",
      `${isOpen ? "关闭" : "打开"}${panelLabel}`,
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
  const analysisState = workbench.querySelector("[data-analysis-state]");
  const actualCharCount = workbench.querySelector(
    "[data-actual-char-count]",
  );
  const chatForm = workbench.querySelector("[data-chat-form]");
  let startAssistantStream = null;
  let autosaveTimer = 0;
  let saveStatusHideTimer = 0;
  let savePromise = null;
  let lastSavedContent = manuscript ? manuscript.value : "";

  function setSaveStatus(message, state = "", hideAfter = 0) {
    if (!saveStatus) return;
    window.clearTimeout(saveStatusHideTimer);
    saveStatus.hidden = !message;
    if (analysisState) analysisState.hidden = Boolean(message);
    saveStatus.textContent = message;
    saveStatus.dataset.state = state;
    if (!hideAfter) return;
    saveStatusHideTimer = window.setTimeout(() => {
      if (saveStatus.dataset.state !== state) return;
      saveStatus.hidden = true;
      saveStatus.textContent = "";
      saveStatus.dataset.state = "";
      if (analysisState) analysisState.hidden = false;
    }, hideAfter);
  }

  function refreshVersionReference(html) {
    if (!html) return;
    const parsed = new DOMParser().parseFromString(html, "text/html");
    const versionId = parsed.querySelector(
      '[data-chat-form] input[name="source_version_id"]',
    );
    const sourceHash = parsed.querySelector(
      '[data-chat-form] input[name="source_hash"]',
    );
    if (chatForm && versionId && sourceHash) {
      chatForm.elements.source_version_id.value = versionId.value;
      chatForm.elements.source_hash.value = sourceHash.value;
    }
    const refreshedAnalysisState = parsed.querySelector(
      "[data-analysis-state]",
    );
    if (analysisState && refreshedAnalysisState) {
      analysisState.textContent =
        refreshedAnalysisState.textContent.trim();
      const refreshedHref =
        refreshedAnalysisState.getAttribute("href");
      if (refreshedHref) {
        analysisState.setAttribute("href", refreshedHref);
      } else {
        analysisState.removeAttribute("href");
      }
    }
  }

  async function saveManuscript(force = false) {
    if (!autosaveForm || !manuscript) return true;
    if (manuscript.value === lastSavedContent) {
      if (force) setSaveStatus("已保存", "saved", 1400);
      return true;
    }
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
          setSaveStatus("已保存", "saved", 1400);
        } else {
          setSaveStatus("未保存", "dirty");
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
      if (actualCharCount) {
        actualCharCount.textContent =
          `${Array.from(manuscript.value).length} 字`;
      }
      setSaveStatus("未保存", "dirty");
      window.clearTimeout(autosaveTimer);
      autosaveTimer = window.setTimeout(() => saveManuscript(), 1200);
    });

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
    const qualityMode = chatForm.querySelector("[data-quality-mode]");
    const chatScroll = workbench.querySelector("[data-chat-scroll]");
    const sendButton = chatForm.querySelector('[type="submit"]');
    let savedQualityMode = qualityMode?.value || "standard";
    let activeEventSource = null;

    const scrollChatToBottom = () => {
      if (!chatScroll) return;
      chatScroll.scrollTop = chatScroll.scrollHeight;
    };

    const setChatBusy = (busy) => {
      if (sendButton) sendButton.disabled = busy;
      if (qualityMode) qualityMode.disabled = busy;
      chatForm.dataset.submitting = busy ? "true" : "false";
    };

    const appendPendingMessages = (question, messageId, quotedText) => {
      if (!chatScroll) return null;
      chatScroll.querySelector(".studio-chat-empty")?.remove();

      const userMessage = document.createElement("article");
      userMessage.className = "studio-chat-message user";
      const userLabel = document.createElement("small");
      userLabel.textContent = "你";
      userMessage.append(userLabel);
      if (quotedText) {
        const quote = document.createElement("blockquote");
        quote.textContent = quotedText;
        userMessage.append(quote);
      }
      const userContent = document.createElement("p");
      userContent.textContent = question;
      userMessage.append(userContent);

      const assistantMessage = document.createElement("article");
      assistantMessage.className = "studio-chat-message assistant streaming";
      assistantMessage.dataset.assistantMessageId = messageId;
      const assistantLabel = document.createElement("small");
      assistantLabel.textContent = "AI";
      const assistantContent = document.createElement("p");
      assistantContent.dataset.streamContent = "";
      assistantContent.textContent = "正在判断任务并读取所需资料……";
      assistantMessage.append(assistantLabel, assistantContent);

      chatScroll.append(userMessage, assistantMessage);
      scrollChatToBottom();
      return assistantMessage;
    };

    const showChatError = (message, assistantMessage = null) => {
      const target =
        assistantMessage?.querySelector("[data-stream-content]") ||
        document.createElement("p");
      target.classList.add("studio-error-text");
      target.textContent = message || "回复失败，请重试。";
      assistantMessage?.classList.remove("streaming");
      if (!assistantMessage && chatScroll) {
        const state = document.createElement("p");
        state.className = "studio-chat-state studio-error-text";
        state.textContent = target.textContent;
        chatScroll.append(state);
      }
      scrollChatToBottom();
    };

    startAssistantStream = (
      messageId,
      assistantMessage,
      redirectUrl = window.location.href,
    ) => {
      if (!window.EventSource) return false;
      activeEventSource?.close();
      const target =
        assistantMessage ||
        chatScroll?.querySelector(
          `[data-assistant-message-id="${messageId}"]`,
        );
      const contentNode = target?.querySelector("[data-stream-content]");
      target?.classList.add("streaming");
      let finished = false;
      const source = new EventSource(
        `/api/assistant/messages/${encodeURIComponent(messageId)}/stream`,
      );
      activeEventSource = source;
      source.addEventListener("snapshot", (event) => {
        const data = JSON.parse(event.data);
        if (contentNode && data.content) {
          contentNode.classList.remove("studio-error-text");
          contentNode.textContent = data.content;
          scrollChatToBottom();
        }
        if (data.status === "failed" && contentNode) {
          showChatError(data.error, target);
        }
      });
      source.addEventListener("done", (event) => {
        finished = true;
        source.close();
        const data = JSON.parse(event.data);
        target?.classList.remove("streaming");
        if (data.status === "failed") {
          showChatError(data.error, target);
          setChatBusy(false);
          return;
        }
        window.setTimeout(() => {
          window.location.replace(data.redirect_url || redirectUrl);
        }, 120);
      });
      source.onerror = () => {
        if (finished) return;
        // EventSource reconnects automatically. The existing status endpoint
        // remains available as a fallback after a manual reload.
      };
      return true;
    };

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

    chatForm.addEventListener("submit", async (event) => {
      if (event.defaultPrevented) return;
      event.preventDefault();
      if (
        chatForm.dataset.submitting === "true" ||
        !chatInput?.value.trim()
      ) {
        return;
      }
      const question = chatInput.value.trim();
      setChatBusy(true);
      try {
        if (
          manuscript &&
          autosaveForm &&
          manuscript.value !== lastSavedContent
        ) {
          const saved = await saveManuscript(true);
          if (!saved) throw new Error("正文保存失败，请重试后再发送");
        }
        const formData = new FormData(chatForm);
        formData.set("question", question);
        const quotedText = String(formData.get("quote_text") || "");
        const response = await fetch(chatForm.action, {
          method: "POST",
          body: formData,
          credentials: "same-origin",
          cache: "no-store",
          headers: {"Accept": "application/json"},
        });
        const result = await response.json();
        if (!response.ok) {
          throw new Error(result.error || "消息发送失败");
        }
        const conversationField = chatForm.elements.conversation_id;
        if (conversationField) {
          conversationField.value = result.conversation_id;
        }
        if (qualityMode) {
          qualityMode.dataset.conversationId = result.conversation_id;
        }
        const assistantMessage = appendPendingMessages(
          question,
          result.message_id,
          quotedText,
        );
        chatInput.value = "";
        clearQuote();
        window.history.replaceState(
          {},
          "",
          result.redirect_url,
        );
        if (
          !startAssistantStream(
            result.message_id,
            assistantMessage,
            result.redirect_url,
          )
        ) {
          window.location.replace(result.redirect_url);
        }
      } catch (error) {
        setChatBusy(false);
        showChatError(error.message || "消息发送失败");
      }
    });

    qualityMode?.addEventListener("change", async () => {
      const conversationId = qualityMode.dataset.conversationId;
      const payload = new FormData();
      payload.set("csrf", chatForm.elements.csrf.value);
      payload.set("quality_mode", qualityMode.value);
      qualityMode.disabled = true;
      try {
        const endpoint = conversationId
          ? `/api/assistant/conversations/${encodeURIComponent(conversationId)}/quality-mode`
          : "/api/settings/quality-mode";
        const response = await fetch(
          endpoint,
          {
            method: "POST",
            body: payload,
            credentials: "same-origin",
            cache: "no-store",
            headers: {"Accept": "application/json"},
          },
        );
        const result = await response.json();
        if (!response.ok) {
          throw new Error(result.error || "模型强度保存失败");
        }
        savedQualityMode = result.quality_mode;
        qualityMode.title = conversationId
          ? "模型强度已保存到当前对话"
          : "模型强度已设为新对话默认值";
      } catch (error) {
        qualityMode.value = savedQualityMode;
        qualityMode.title =
          error.message || "模型强度保存失败，请稍后重试";
      } finally {
        qualityMode.disabled = false;
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
    if (startAssistantStream?.(pendingMessageId, null)) return;
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

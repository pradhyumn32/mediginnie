// Medijinn Frontend Application Logic
// Real-time SSE streaming, entity rendering, emergency triage detection, and drug catalog

document.addEventListener("DOMContentLoaded", () => {
  const chatWindow = document.getElementById("chat-window");
  const welcomeHero = document.getElementById("welcome-hero");
  const userInput = document.getElementById("user-input");
  const sendBtn = document.getElementById("send-btn");
  const healthBadge = document.getElementById("health-badge");
  const healthText = document.getElementById("health-text");
  const emergencyBanner = document.getElementById("emergency-banner");
  const emergencyTitle = document.getElementById("emergency-title");
  const emergencyBody = document.getElementById("emergency-body");
  
  // Modal elements
  const catalogModal = document.getElementById("catalog-modal");
  const btnOpenCatalog = document.getElementById("btn-open-catalog");
  const btnCloseModal = document.getElementById("btn-close-modal");
  const btnNewChat = document.getElementById("btn-new-chat");
  const drugTableBody = document.getElementById("drug-table-body");
  const btnSubmitIngest = document.getElementById("btn-submit-ingest");
  const ingestDrugName = document.getElementById("ingest-drug-name");
  const ingestSource = document.getElementById("ingest-source");
  const ingestFeedback = document.getElementById("ingest-feedback");

  let isGenerating = false;
  let chatHistory = [];
  const STORAGE_KEY = "medijinn_chat_history";

  function saveChatHistory() {
    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(chatHistory));
    } catch (e) {
      console.warn("Could not save chat history to sessionStorage:", e);
    }
  }

  function loadChatHistory() {
    try {
      const saved = sessionStorage.getItem(STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed) && parsed.length > 0) {
          chatHistory = parsed;
          if (welcomeHero) welcomeHero.style.display = "none";
          parsed.forEach((m) => {
            if (m.role === "user") {
              appendMessageRow("user", m.content);
            } else if (m.role === "assistant") {
              const row = appendAssistantPlaceholder();
              const bubbleEl = row.querySelector(".message-bubble");
              const contentEl = bubbleEl.querySelector(".bubble-content");
              contentEl.innerHTML = marked.parse(m.content);
            }
          });
          chatWindow.scrollTop = chatWindow.scrollHeight;
        }
      }
    } catch (e) {
      console.warn("Could not load chat history from sessionStorage:", e);
    }
  }

  if (btnNewChat) {
    btnNewChat.addEventListener("click", () => {
      chatHistory = [];
      try {
        sessionStorage.removeItem(STORAGE_KEY);
      } catch {}
      // Clear rendered messages
      const rows = chatWindow.querySelectorAll(".message-row");
      rows.forEach((r) => r.remove());
      if (welcomeHero) welcomeHero.style.display = "flex";
      emergencyBanner.classList.remove("active");
      userInput.value = "";
      userInput.style.height = "auto";
    });
  }

  // Restore existing conversation if present
  loadChatHistory();

  // 1. Initial Health Check & Statistics
  async function refreshHealth() {
    try {
      const res = await fetch("/api/health");
      if (res.ok) {
        const data = await res.json();
        healthText.textContent = `Online | ${data.total_drugs} Drugs | ${data.total_chunks} Chunks`;
      } else {
        healthText.textContent = "Service Degraded";
      }
    } catch {
      healthText.textContent = "Offline";
    }
  }
  refreshHealth();
  setInterval(refreshHealth, 30000);

  // 2. Textarea Auto-Resize
  userInput.addEventListener("input", () => {
    userInput.style.height = "auto";
    userInput.style.height = `${Math.min(userInput.scrollHeight, 140)}px`;
    sendBtn.disabled = !userInput.value.trim() || isGenerating;
  });

  userInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!isGenerating && userInput.value.trim()) {
        sendMessage(userInput.value.trim());
      }
    }
  });

  // 3. Suggested Prompt Cards
  document.querySelectorAll(".prompt-card").forEach((card) => {
    card.addEventListener("click", () => {
      const prompt = card.getAttribute("data-prompt");
      if (prompt && !isGenerating) {
        userInput.value = prompt;
        sendMessage(prompt);
      }
    });
  });

  sendBtn.addEventListener("click", () => {
    const text = userInput.value.trim();
    if (text && !isGenerating) {
      sendMessage(text);
    }
  });

  // 4. Send Message & Stream Response
  async function sendMessage(text) {
    isGenerating = true;
    sendBtn.disabled = true;
    userInput.value = "";
    userInput.style.height = "auto";

    if (welcomeHero) {
      welcomeHero.style.display = "none";
    }
    emergencyBanner.classList.remove("active");

    // History to send to the backend includes all prior conversation turns (up to 20 messages)
    const priorHistory = chatHistory.slice(-20);

    // Append User Message
    appendMessageRow("user", text);
    chatHistory.push({ role: "user", content: text });
    saveChatHistory();

    // Append Assistant Placeholder Bubble
    const assistantRow = appendAssistantPlaceholder();
    const bubbleEl = assistantRow.querySelector(".message-bubble");
    const contentEl = bubbleEl.querySelector(".bubble-content");
    const entitiesEl = bubbleEl.querySelector(".entities-badge-bar");
    const sourcesEl = bubbleEl.querySelector(".sources-container");

    try {
      const response = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          chat_history: priorHistory,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let fullMessage = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split("\n\n");
        buffer = events.pop(); // keep last incomplete event in buffer

        for (const evt of events) {
          if (!evt.trim()) continue;

          let eventName = "message";
          let dataStr = "";

          const lines = evt.split("\n");
          for (const line of lines) {
            if (line.startsWith("event: ")) {
              eventName = line.substring(7).trim();
            } else if (line.startsWith("data: ")) {
              dataStr = line.substring(6).trim();
            }
          }

          if (!dataStr) continue;
          let parsedData = {};
          try {
            parsedData = JSON.parse(dataStr);
          } catch {
            continue;
          }

          // Handle SSE event types
          if (eventName === "triage") {
            if (parsedData.is_emergency) {
              triggerEmergencyAlert(parsedData.emergency_message);
            }
          } else if (eventName === "entities") {
            renderEntityBadges(entitiesEl, parsedData);
          } else if (eventName === "sources") {
            renderSourcesAccordion(sourcesEl, parsedData.sources);
          } else if (eventName === "message") {
            fullMessage = parsedData.response || "";
            contentEl.innerHTML = marked.parse(fullMessage);
            chatWindow.scrollTop = chatWindow.scrollHeight;
          }
        }
      }

      if (fullMessage) {
        chatHistory.push({ role: "assistant", content: fullMessage });
        saveChatHistory();
      }

    } catch (err) {
      console.warn("SSE stream interrupted or failed, falling back to JSON /api/chat:", err);
      // Fallback to synchronous endpoint
      try {
        const fallbackRes = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message: text,
            chat_history: priorHistory,
          }),
        });
        const data = await fallbackRes.json();
        
        if (data.is_emergency) {
          triggerEmergencyAlert(data.emergency_message);
        }

        renderEntityBadges(entitiesEl, {
          canonical_drugs: data.canonical_drugs,
          intent: data.intent,
        });

        if (data.sources && data.sources.length) {
          renderSourcesAccordion(sourcesEl, data.sources);
        }

        contentEl.innerHTML = marked.parse(data.response || "No response received.");
        chatHistory.push({ role: "assistant", content: data.response });
        saveChatHistory();

      } catch (fallbackErr) {
        contentEl.innerHTML = `<p style="color: #f87171;">⚠️ Error connecting to Medijinn backend: ${fallbackErr.message}</p>`;
      }
    } finally {
      isGenerating = false;
      sendBtn.disabled = !userInput.value.trim();
      chatWindow.scrollTop = chatWindow.scrollHeight;
    }
  }

  // 5. DOM Helpers
  function appendMessageRow(role, text) {
    const row = document.createElement("div");
    row.className = `message-row ${role}`;
    row.innerHTML = `
      <div class="avatar ${role}">${role === "user" ? "You" : "MJ"}</div>
      <div class="message-bubble">
        <p>${escapeHtml(text)}</p>
      </div>
    `;
    chatWindow.appendChild(row);
    chatWindow.scrollTop = chatWindow.scrollHeight;
  }

  function appendAssistantPlaceholder() {
    const row = document.createElement("div");
    row.className = "message-row assistant";
    row.innerHTML = `
      <div class="avatar assistant">MJ</div>
      <div class="message-bubble glass">
        <div class="entities-badge-bar" style="display: none;"></div>
        <div class="bubble-content">
          <div class="typing-indicator">
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
          </div>
        </div>
        <div class="sources-container" style="display: none;"></div>
      </div>
    `;
    chatWindow.appendChild(row);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    return row;
  }

  function renderEntityBadges(container, data) {
    const drugs = data.canonical_drugs || [];
    const intent = data.intent || "";

    if (!drugs.length && !intent) return;
    container.style.display = "flex";
    container.innerHTML = "";

    drugs.forEach((d) => {
      const pill = document.createElement("span");
      pill.className = "entity-pill drug";
      pill.textContent = `💊 ${d}`;
      container.appendChild(pill);
    });

    if (intent) {
      const pill = document.createElement("span");
      pill.className = "entity-pill intent";
      pill.textContent = `🎯 ${intent.replace("_", " ")}`;
      container.appendChild(pill);
    }
  }

  function renderSourcesAccordion(container, sources) {
    if (!sources || !sources.length) return;
    container.style.display = "block";
    container.innerHTML = `
      <button class="sources-toggle" type="button">
        <span>📚 Verified Clinical Sources (${sources.length})</span>
        <span>▼</span>
      </button>
      <div class="sources-list" style="display: none;"></div>
    `;

    const toggleBtn = container.querySelector(".sources-toggle");
    const listEl = container.querySelector(".sources-list");

    toggleBtn.addEventListener("click", () => {
      const isHidden = listEl.style.display === "none";
      listEl.style.display = isHidden ? "flex" : "none";
      toggleBtn.querySelector("span:last-child").textContent = isHidden ? "▲" : "▼";
    });

    sources.forEach((s) => {
      const card = document.createElement("div");
      card.className = "source-card";
      const sev = s.severity || "INFO";
      const sec = s.section || s.section_title || "Label Section";
      const drug = s.drug || s.generic_name || "Drug";
      const src = (s.source || "FDA").toUpperCase();

      card.innerHTML = `
        <div>
          <div class="source-title">${escapeHtml(drug)} &mdash; ${escapeHtml(sec)}</div>
          <span class="source-tag">Source: ${src}</span>
        </div>
        <div class="source-badges">
          <span class="badge ${sev}">${sev.replace("_", " ")}</span>
        </div>
      `;
      listEl.appendChild(card);
    });
  }

  function triggerEmergencyAlert(msg) {
    emergencyBanner.classList.add("active");
    if (msg) {
      emergencyBody.innerHTML = marked.parse(msg);
    }
    chatWindow.scrollTop = 0;
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.innerText = str;
    return div.innerHTML;
  }

  // 6. Drug Catalog Modal & Ingestion
  btnOpenCatalog.addEventListener("click", () => {
    catalogModal.classList.add("active");
    loadDrugCatalog();
  });

  btnCloseModal.addEventListener("click", () => {
    catalogModal.classList.remove("active");
  });

  catalogModal.addEventListener("click", (e) => {
    if (e.target === catalogModal) {
      catalogModal.classList.remove("active");
    }
  });

  async function loadDrugCatalog() {
    drugTableBody.innerHTML = `<tr><td colspan="3" style="text-align: center; color: var(--text-muted);">Fetching catalog from pgvector...</td></tr>`;
    try {
      const res = await fetch("/api/ingest/drugs");
      const data = await res.json();

      if (!data.length) {
        drugTableBody.innerHTML = `<tr><td colspan="3" style="text-align: center; color: var(--text-muted);">No drugs ingested yet. Use the form above to ingest.</td></tr>`;
        return;
      }

      drugTableBody.innerHTML = "";
      data.forEach((d) => {
        const tr = document.createElement("tr");
        const brands = d.brand_names || [];
        const brandsHtml = brands.length
          ? brands.slice(0, 6).map((b) => `<span class="brand-mini-pill">${escapeHtml(b)}</span>`).join(" ") + (brands.length > 6 ? ` <span style="font-size: 0.7rem; color: var(--text-muted);">+${brands.length - 6} more</span>` : "")
          : `<span style="color: var(--text-muted); font-size: 0.8rem;">None recorded</span>`;

        tr.innerHTML = `
          <td><strong>${escapeHtml(d.generic_name)}</strong></td>
          <td><div class="brand-pill-list">${brandsHtml}</div></td>
          <td><span class="badge INFO">${d.chunk_count} chunks</span></td>
        `;
        drugTableBody.appendChild(tr);
      });
    } catch (err) {
      drugTableBody.innerHTML = `<tr><td colspan="3" style="text-align: center; color: #f87171;">Failed to load catalog: ${err.message}</td></tr>`;
    }
  }

  btnSubmitIngest.addEventListener("click", async () => {
    const drugName = ingestDrugName.value.trim();
    const source = ingestSource.value;

    if (!drugName) {
      alert("Please enter a medication name.");
      return;
    }

    btnSubmitIngest.disabled = true;
    btnSubmitIngest.textContent = "Ingesting & Embedding...";
    ingestFeedback.style.display = "block";
    ingestFeedback.style.color = "var(--accent-cyan)";
    ingestFeedback.textContent = `Resolving ${drugName} via RxNorm and fetching from ${source.toUpperCase()}...`;

    try {
      const res = await fetch("/api/ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          drug_name: drugName,
          source: source,
          max_records: 1,
        }),
      });

      const data = await res.json();
      if (!res.ok || data.status !== "success") {
        throw new Error(data.detail || data.error || "Ingestion failed");
      }

      ingestFeedback.style.color = "#10b981";
      ingestFeedback.textContent = `✓ Successfully ingested ${data.drug} (${data.chunks_ingested} chunks embedded)!`;
      ingestDrugName.value = "";
      loadDrugCatalog();
      refreshHealth();
    } catch (err) {
      ingestFeedback.style.color = "#f87171";
      ingestFeedback.textContent = `✗ Ingestion Error: ${err.message}`;
    } finally {
      btnSubmitIngest.disabled = false;
      btnSubmitIngest.textContent = "Ingest Label";
    }
  });

});

/**
 * Enterprise Agentic RAG - Frontend Application Logic
 * Supports DOCX, PDF, Markdown & Plaintext uploads, live ReAct trajectory streaming (SSE),
 * and interactive citation inspection.
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Element References
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const browseBtn = document.getElementById("browse-btn");
    const uploadProgressContainer = document.getElementById("upload-progress-container");
    const uploadProgressFill = document.getElementById("upload-progress-fill");
    const uploadStatusText = document.getElementById("upload-status-text");
    const documentList = document.getElementById("document-list");
    const refreshStatsBtn = document.getElementById("refresh-stats-btn");

    const statChunksCount = document.getElementById("stat-chunks-count");
    const statGraphCount = document.getElementById("stat-graph-count");
    const activeEngineText = document.getElementById("active-engine-text");

    const chatFeed = document.getElementById("chat-feed");
    const heroWelcome = document.getElementById("hero-welcome");
    const chatForm = document.getElementById("chat-form");
    const queryInput = document.getElementById("query-input");
    const sendBtn = document.getElementById("send-btn");
    const toastContainer = document.getElementById("toast-container");

    // Theme Switcher (Dark Velvet Pink / Clean White)
    const themeToggleBtn = document.getElementById("theme-toggle-btn");
    const themeToggleLabel = document.getElementById("theme-toggle-label");
    const themeIconDark = document.querySelector(".theme-icon-dark");
    const themeIconLight = document.querySelector(".theme-icon-light");

    function applyTheme(theme) {
        document.documentElement.setAttribute("data-theme", theme);
        localStorage.setItem("rag_theme", theme);
        if (theme === "light") {
            if (themeIconDark) themeIconDark.style.display = "none";
            if (themeIconLight) themeIconLight.style.display = "block";
            if (themeToggleLabel) themeToggleLabel.textContent = "Dark Velvet";
        } else {
            if (themeIconDark) themeIconDark.style.display = "block";
            if (themeIconLight) themeIconLight.style.display = "none";
            if (themeToggleLabel) themeToggleLabel.textContent = "Clean White";
        }
    }

    const savedTheme = localStorage.getItem("rag_theme") || "dark";
    applyTheme(savedTheme);

    if (themeToggleBtn) {
        themeToggleBtn.addEventListener("click", () => {
            const currentTheme = document.documentElement.getAttribute("data-theme") || "dark";
            const nextTheme = currentTheme === "dark" ? "light" : "dark";
            applyTheme(nextTheme);
            showToast(`Theme switched to ${nextTheme === 'light' ? 'Clean White & Dark Pink' : 'Dark Velvet Pink & White'}`, "info");
        });
    }

    // ==========================================================================
    // 0. API Key Handling
    // ==========================================================================
    // The key is never bundled with the app: the operator enters it, it lives in
    // sessionStorage, and it is attached as X-API-Key on every request.
    const authGate = document.getElementById("auth-gate");
    const authForm = document.getElementById("auth-form");
    const authInput = document.getElementById("auth-key-input");
    const authError = document.getElementById("auth-error");
    const authCancelBtn = document.getElementById("auth-cancel");
    const authKeyBtn = document.getElementById("auth-key-btn");
    const authKeyLabel = document.getElementById("auth-key-label");

    let apiKey = sessionStorage.getItem("rag_api_key") || "";

    function updateKeyBadge() {
        if (authKeyLabel) authKeyLabel.textContent = apiKey ? "Key Set" : "API Key";
        if (authKeyBtn) authKeyBtn.classList.toggle("key-active", Boolean(apiKey));
    }

    function showAuthGate(message) {
        if (!authGate) return;
        if (message) {
            authError.textContent = message;
            authError.style.display = "block";
        } else {
            authError.style.display = "none";
        }
        authGate.style.display = "flex";
        authInput.value = "";
        authInput.focus();
    }

    function hideAuthGate() {
        if (authGate) authGate.style.display = "none";
    }

    function extractErrorMessage(payload, fallback) {
        if (!payload) return fallback;
        if (payload.error && payload.error.message) return payload.error.message;
        if (typeof payload.detail === "string") return payload.detail;
        return fallback;
    }

    async function authFetch(url, options = {}) {
        const headers = new Headers(options.headers || {});
        if (apiKey) headers.set("X-API-Key", apiKey);
        const response = await fetch(url, { ...options, headers });
        if (response.status === 401) {
            const payload = await response.clone().json().catch(() => null);
            showAuthGate(
                apiKey
                    ? "That API key was rejected. Enter a valid key."
                    : extractErrorMessage(payload, "An API key is required.")
            );
        }
        return response;
    }

    if (authKeyBtn) authKeyBtn.addEventListener("click", () => showAuthGate(""));
    if (authCancelBtn) authCancelBtn.addEventListener("click", hideAuthGate);
    if (authForm) {
        authForm.addEventListener("submit", (event) => {
            event.preventDefault();
            const candidate = authInput.value.trim();
            if (!candidate) return;
            apiKey = candidate;
            sessionStorage.setItem("rag_api_key", apiKey);
            updateKeyBadge();
            hideAuthGate();
            showToast("API key stored for this session", "success");
            fetchSystemStats();
            fetchDocuments();
        });
    }
    updateKeyBadge();

    // Local Session State
    let uploadedFiles = [];
    let serverDocuments = [];
    let isProcessing = false;

    // ==========================================================================
    // 1. Toast Notification System
    // ==========================================================================
    function showToast(message, type = "info") {
        const toast = document.createElement("div");
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `
            <span>${message}</span>
        `;
        toastContainer.appendChild(toast);
        setTimeout(() => {
            toast.style.opacity = "0";
            toast.style.transform = "translateX(40px)";
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }

    // ==========================================================================
    // 2. Fetch System & Agent Status
    // ==========================================================================
    async function fetchSystemStats() {
        try {
            // /ready is the probe that reports index size; /health is liveness only.
            // Keep the authenticated probe quiet until a key exists, so a first
            // visit does not open the key gate before the user does anything.
            const agentPromise = apiKey
                ? authFetch("/api/v1/agent/info").then(r => (r.ok ? r.json() : null)).catch(() => null)
                : Promise.resolve(null);

            const [readyRes, agentRes] = await Promise.all([
                fetch("/ready").then(r => (r.ok ? r.json() : null)).catch(() => null),
                agentPromise
            ]);

            if (readyRes) {
                statChunksCount.textContent = readyRes.child_chunks || 0;
                statGraphCount.textContent = readyRes.graph_nodes || 0;
            }

            if (agentRes) {
                const prov = (agentRes.provider || "local").toUpperCase();
                const model = agentRes.model || "deterministic";
                activeEngineText.textContent = `${prov}: ${model}`;
            }
        } catch (err) {
            console.error("Failed to load metrics:", err);
        }
    }

    // Initial stats load
    fetchSystemStats();
    refreshStatsBtn.addEventListener("click", () => {
        fetchSystemStats();
        fetchDocuments();
        showToast("System metrics refreshed", "info");
    });

    // ==========================================================================
    // 3. Document Ingestion & Drag-and-Drop (PDF, DOCX, MD, TXT)
    // ==========================================================================
    browseBtn.addEventListener("click", () => fileInput.click());
    dropZone.addEventListener("click", (e) => {
        if (e.target !== browseBtn) fileInput.click();
    });

    ["dragenter", "dragover"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add("drag-active");
        });
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove("drag-active");
        });
    });

    dropZone.addEventListener("drop", (e) => {
        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            handleFileUploads(Array.from(files));
        }
    });

    fileInput.addEventListener("change", (e) => {
        const files = e.target.files;
        if (files && files.length > 0) {
            handleFileUploads(Array.from(files));
        }
    });

    async function handleFileUploads(files) {
        if (!files.length) return;

        const allowedExtensions = [".pdf", ".docx", ".md", ".markdown", ".txt", ".rst"];
        const validFiles = files.filter(f => {
            const ext = "." + f.name.split(".").pop().toLowerCase();
            return allowedExtensions.includes(ext);
        });

        if (validFiles.length === 0) {
            showToast("Unsupported file type. Please upload PDF, DOCX, Markdown, or TXT files.", "error");
            return;
        }

        uploadProgressContainer.style.display = "block";
        const total = validFiles.length;

        for (let i = 0; i < total; i++) {
            const file = validFiles[i];
            const pct = Math.round(((i + 1) / total) * 100);
            uploadProgressFill.style.width = `${pct}%`;
            uploadStatusText.textContent = `Indexing ${file.name} (${i + 1} of ${total})...`;

            const formData = new FormData();
            formData.append("file", file);

            try {
                const res = await authFetch("/api/v1/ingest/file", {
                    method: "POST",
                    body: formData
                });
                const result = await res.json().catch(() => null);

                if (res.ok && result && result.status === "success") {
                    const ext = file.name.split(".").pop().toUpperCase();
                    uploadedFiles.unshift({
                        name: file.name,
                        ext: ext,
                        chunks: result.data.child_chunks_count,
                        graphNodes: result.data.total_indexed_nodes
                    });
                    renderUploadedList();
                    showToast(`Successfully indexed ${file.name}!`, "success");
                } else {
                    showToast(
                        `Failed to index ${file.name}: ${extractErrorMessage(result, `HTTP ${res.status}`)}`,
                        "error"
                    );
                }
            } catch (err) {
                showToast(`Network error uploading ${file.name}`, "error");
            }
        }

        setTimeout(() => {
            uploadProgressContainer.style.display = "none";
            uploadProgressFill.style.width = "0%";
            fileInput.value = "";
            fetchSystemStats();
            fetchDocuments();
        }, 800);
    }

    // ------------------------------------------------------------------
    // Document library (server-backed, so it survives a page reload)
    // ------------------------------------------------------------------
    async function fetchDocuments() {
        if (!apiKey) return;
        try {
            const res = await authFetch("/api/v1/documents");
            if (!res.ok) return;
            const payload = await res.json();
            serverDocuments = payload.documents || [];
            renderUploadedList();
        } catch (err) {
            console.error("Failed to load document library:", err);
        }
    }

    function docBadge(title) {
        const match = /\.([a-z0-9]{2,5})$/i.exec(title || "");
        return match ? match[1].toUpperCase() : "DOC";
    }

    function renderUploadedList() {
        const library = serverDocuments.length
            ? serverDocuments.map(doc => ({
                name: doc.title || "Untitled Document",
                ext: docBadge(doc.title),
                chunks: doc.child_chunks || 0
            }))
            : uploadedFiles;

        if (library.length === 0) {
            documentList.innerHTML = `
                <div class="empty-list-placeholder">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
                        <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"></path>
                        <polyline points="13 2 13 9 20 9"></polyline>
                    </svg>
                    <span>No documents indexed yet.</span>
                </div>
            `;
            return;
        }

        documentList.innerHTML = library.map(doc => `
            <div class="doc-item">
                <div class="doc-item-title" title="${escapeHtml(doc.name)}">${escapeHtml(doc.name)}</div>
                <div class="doc-item-meta">
                    <span class="doc-type-badge">${escapeHtml(doc.ext)}</span>
                    <span style="font-size: 0.72rem; color: var(--text-muted);">${doc.chunks} ch</span>
                </div>
            </div>
        `).join("");
    }

    fetchDocuments();

    // ==========================================================================
    // 4. Chat, Auto-Resizing & Sample Prompts
    // ==========================================================================
    queryInput.addEventListener("input", () => {
        queryInput.style.height = "auto";
        queryInput.style.height = `${Math.min(queryInput.scrollHeight, 140)}px`;
    });

    queryInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event("submit"));
        }
    });

    document.querySelectorAll(".sample-prompt-pill").forEach(pill => {
        pill.addEventListener("click", () => {
            const promptText = pill.dataset.prompt;
            queryInput.value = promptText;
            queryInput.dispatchEvent(new Event("input"));
            queryInput.focus();
        });
    });

    // ==========================================================================
    // 5. Server-Sent Events (SSE) Agent Reasoning & Streaming
    // ==========================================================================
    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const query = queryInput.value.trim();
        if (!query || isProcessing) return;

        // Hide welcome banner on first query
        if (heroWelcome) heroWelcome.style.display = "none";

        appendUserMessage(query);

        queryInput.value = "";
        queryInput.style.height = "auto";
        isProcessing = true;
        sendBtn.disabled = true;

        // Create Agent container with Trajectory Card and Answer Placeholder
        const agentContainer = createAgentMessageContainer();
        chatFeed.appendChild(agentContainer.row);
        scrollToBottom();

        try {
            await streamAgentQuery(query, agentContainer);
        } catch (err) {
            console.error("Agent error:", err);
            agentContainer.answerBody.innerHTML = `<span style="color: var(--danger)">Agent execution encountered an error: ${err.message}</span>`;
        } finally {
            isProcessing = false;
            sendBtn.disabled = false;
            queryInput.focus();
            scrollToBottom();
        }
    });

    function appendUserMessage(text) {
        const row = document.createElement("div");
        row.className = "message-row user-row";
        row.innerHTML = `
            <div class="message-avatar user-avatar">YOU</div>
            <div class="message-bubble">
                <div class="user-bubble-content">${escapeHtml(text)}</div>
            </div>
        `;
        chatFeed.appendChild(row);
        scrollToBottom();
    }

    function createAgentMessageContainer() {
        const row = document.createElement("div");
        row.className = "message-row agent-row";

        row.innerHTML = `
            <div class="message-avatar agent-avatar">AI</div>
            <div class="message-bubble">
                <!-- Trajectory Telemetry Card -->
                <div class="agent-trajectory-card">
                    <div class="trajectory-header">
                        <div class="trajectory-header-left">
                            <div class="trajectory-spinner" id="traj-spinner"></div>
                            <span class="trajectory-title">
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
                                    <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
                                </svg>
                                Autonomous ReAct Trajectory
                            </span>
                        </div>
                        <span class="trajectory-toggle-btn">▼</span>
                    </div>
                    <div class="trajectory-body">
                        <div class="plan-container" style="display: none;">
                            <div class="plan-title">Planned Sub-Goals</div>
                            <ol class="plan-list"></ol>
                        </div>
                        <div class="steps-container"></div>
                    </div>
                </div>

                <!-- Grounded Answer Card -->
                <div class="agent-answer-card">
                    <div class="answer-body">
                        <span style="color: var(--text-muted); font-style: italic;">Reasoning and retrieving evidence...</span>
                    </div>
                    <div class="unsupported-claims" style="display: none;"></div>
                    <div class="answer-footer" style="display: none;">
                        <div class="verification-tag verified">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                                <polyline points="20 6 9 17 4 12"></polyline>
                            </svg>
                            <span class="verif-status">VERIFIED</span>
                        </div>
                        <div class="metrics-row">
                            <span class="metric-faithfulness">Faithfulness: 1.00</span>
                            <span class="metric-latency">0ms</span>
                        </div>
                    </div>
                </div>
            </div>
        `;

        const header = row.querySelector(".trajectory-header");
        const body = row.querySelector(".trajectory-body");
        const toggleBtn = row.querySelector(".trajectory-toggle-btn");
        header.addEventListener("click", () => {
            const isHidden = body.style.display === "none";
            body.style.display = isHidden ? "flex" : "none";
            toggleBtn.textContent = isHidden ? "▼" : "▲";
        });

        return {
            row,
            spinner: row.querySelector("#traj-spinner"),
            planContainer: row.querySelector(".plan-container"),
            planList: row.querySelector(".plan-list"),
            stepsContainer: row.querySelector(".steps-container"),
            answerBody: row.querySelector(".answer-body"),
            claimsBox: row.querySelector(".unsupported-claims"),
            answerFooter: row.querySelector(".answer-footer"),
            verifStatus: row.querySelector(".verif-status"),
            metricFaithfulness: row.querySelector(".metric-faithfulness"),
            metricLatency: row.querySelector(".metric-latency")
        };
    }

    async function streamAgentQuery(query, container) {
        // Use Fetch with Streaming Reader for SSE endpoint
        const response = await authFetch("/api/v1/agent/stream", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: query })
        });

        if (response.status === 401) {
            if (container.spinner) container.spinner.style.display = "none";
            container.answerBody.textContent = "An API key is required to query the index.";
            return;
        }

        if (!response.ok) {
            // Fallback to the synchronous agent query (same auth, richer errors)
            const fallbackRes = await authFetch("/api/v1/agent/query", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query: query })
            });
            const data = await fallbackRes.json().catch(() => null);
            if (!fallbackRes.ok || !data) {
                if (container.spinner) container.spinner.style.display = "none";
                container.answerBody.innerHTML = `<span style="color: var(--danger)">${escapeHtml(
                    extractErrorMessage(data, `Request failed (HTTP ${fallbackRes.status}).`)
                )}</span>`;
                return;
            }
            renderFinalAnswer(data, container);
            return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const events = buffer.split("\n\n");
            buffer = events.pop(); // Retain incomplete chunk

            for (const rawEvent of events) {
                if (!rawEvent.trim()) continue;
                parseAndHandleSSE(rawEvent, container);
            }
            scrollToBottom();
        }
    }

    function parseAndHandleSSE(rawEvent, container) {
        let eventType = "message";
        let dataStr = "";

        const lines = rawEvent.split("\n");
        for (const line of lines) {
            if (line.startsWith("event:")) {
                eventType = line.replace("event:", "").trim().toLowerCase();
            } else if (line.startsWith("data:")) {
                dataStr = line.replace("data:", "").trim();
            }
        }

        if (dataStr === "[DONE]") {
            if (container.spinner) container.spinner.style.display = "none";
            return;
        }

        try {
            const data = JSON.parse(dataStr);

            switch (eventType) {
                case "on_plan":
                    if (data.subgoals && data.subgoals.length) {
                        container.planContainer.style.display = "block";
                        container.planList.innerHTML = data.subgoals.map(g => `<li>${escapeHtml(g)}</li>`).join("");
                    }
                    break;

                case "on_thought":
                    addStepElement(container.stepsContainer, "thought", `🤔 Step ${data.step}: ${data.thought}`);
                    break;

                case "on_action":
                    const argsFormatted = JSON.stringify(data.args || {});
                    addStepElement(container.stepsContainer, "action", `⚙️ Calling tool: <span class="action-tool-badge">${data.tool}</span>(${escapeHtml(argsFormatted)})`);
                    break;

                case "on_observation":
                    const isRel = data.is_relevant;
                    const badgeClass = isRel ? "relevant" : "low";
                    const badgeText = isRel ? "✓ Relevant" : "⚠️ Low Relevance";
                    let obsPreview = typeof data.observation === "object" ? JSON.stringify(data.observation) : String(data.observation);
                    if (obsPreview.length > 250) obsPreview = obsPreview.substring(0, 250) + "...";

                    addStepElement(container.stepsContainer, "observation", `
                        <div class="observation-header">
                            <strong>Tool Output</strong>
                            <span class="relevance-pill ${badgeClass}">${badgeText}</span>
                        </div>
                        <div>${escapeHtml(obsPreview)}</div>
                    `);
                    break;

                case "on_answer":
                    renderFinalAnswer(data, container);
                    break;
            }
        } catch (e) {
            console.error("SSE parse error:", e, rawEvent);
        }
    }

    function addStepElement(stepsContainer, type, contentHtml) {
        const stepDiv = document.createElement("div");
        stepDiv.className = `trajectory-step step-${type}`;
        stepDiv.innerHTML = contentHtml;
        stepsContainer.appendChild(stepDiv);
    }

    function renderFinalAnswer(data, container) {
        if (container.spinner) container.spinner.style.display = "none";

        const rawAnswer = data.answer || "No response produced.";

        // Highlight bracket citations [doc#chunk] as interactive chips
        container.answerBody.innerHTML = formatMarkdownWithCitations(rawAnswer);

        container.answerFooter.style.display = "flex";
        const faith = typeof data.faithfulness === "number"
            ? data.faithfulness
            : (typeof data.faithfulness_score === "number" ? data.faithfulness_score : 0);
        const latency = typeof data.latency_ms === "number" ? `${Math.round(data.latency_ms)}ms` : "n/a";

        container.metricFaithfulness.textContent = `Faithfulness: ${Number(faith).toFixed(2)}`;
        container.metricLatency.textContent = latency;

        // Trust the server's verdict, not a phrase match on the answer text.
        const verified = data.verification_passed === true;
        const abstained = data.abstained === true;
        const tag = container.verifStatus.parentElement;

        if (abstained) {
            // The system declined to answer, so there are no claims to grade.
            // Rendering a faithfulness score and a claim list here read as "the
            // answer scored badly", when in fact no answer was produced.
            container.metricFaithfulness.textContent = "Faithfulness: n/a";
            container.verifStatus.textContent = "ABSTAINED — NO SUPPORTING EVIDENCE";
            tag.className = "verification-tag abstained";
            renderUnsupportedClaims(container, []);
            return;
        }

        container.verifStatus.textContent = verified ? "VERIFIED GROUNDED" : "UNVERIFIED";
        tag.className = verified ? "verification-tag verified" : "verification-tag abstained";

        renderUnsupportedClaims(container, data.unsupported_claims);
    }

    // Claim-level evidence: show exactly which sentences the critic could not
    // tie to a retrieved passage, instead of a single opaque badge.
    function renderUnsupportedClaims(container, claims) {
        const box = container.claimsBox;
        if (!box) return;

        const items = Array.isArray(claims) ? claims.filter(Boolean) : [];
        if (items.length === 0) {
            box.style.display = "none";
            box.innerHTML = "";
            return;
        }

        box.style.display = "block";
        box.innerHTML = `
            <div class="claims-header">
                <strong>${items.length} claim${items.length > 1 ? "s" : ""} not backed by a retrieved source</strong>
                <span>Treat these statements as unverified.</span>
            </div>
            ${items.map(claim => `
                <div class="claim-item">
                    <span class="claim-badge">Unsupported</span>
                    <p>${escapeHtml(claim)}</p>
                </div>
            `).join("")}
        `;
    }

    // ==========================================================================
    // 6. Interactive Citation Inspection Modal
    // ==========================================================================
    const citationModal = document.getElementById("citation-modal");
    const modalChunkTitle = document.getElementById("modal-chunk-title");
    const modalCloseBtn = document.getElementById("modal-close-btn");
    const modalCitationTag = document.getElementById("modal-citation-tag");
    const modalChunkText = document.getElementById("modal-chunk-text");
    const modalParentContainer = document.getElementById("modal-parent-container");
    const modalParentText = document.getElementById("modal-parent-text");

    function closeModal() {
        if (citationModal) citationModal.style.display = "none";
    }

    if (modalCloseBtn) modalCloseBtn.addEventListener("click", closeModal);
    if (citationModal) {
        citationModal.addEventListener("click", (e) => {
            if (e.target === citationModal) closeModal();
        });
    }

    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && citationModal && citationModal.style.display !== "none") {
            closeModal();
        }
    });

    async function openCitationModal(tag) {
        if (!citationModal) return;
        citationModal.style.display = "flex";
        modalChunkTitle.textContent = "Loading source...";
        modalCitationTag.textContent = `[${tag}]`;
        modalChunkText.textContent = "Fetching verified evidence passage...";
        modalParentContainer.style.display = "none";

        try {
            const res = await authFetch(`/api/v1/chunk/${encodeURIComponent(tag)}`);
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                modalChunkTitle.textContent = "Evidence Chunk";
                modalChunkText.textContent = extractErrorMessage(
                    err,
                    `Unable to load chunk details for [${tag}].`
                );
                return;
            }

            const data = await res.json();
            modalChunkTitle.textContent = data.title || "Indexed Document";
            modalCitationTag.textContent = data.citation_tag || `[${tag}]`;
            modalChunkText.textContent = data.text || "No content found in chunk.";

            if (data.parent_text && data.parent_text.trim() !== data.text.trim()) {
                modalParentContainer.style.display = "block";
                modalParentText.textContent = data.parent_text;
            } else {
                modalParentContainer.style.display = "none";
            }
        } catch (err) {
            modalChunkTitle.textContent = "Error";
            modalChunkText.textContent = `Failed to connect to chunk API: ${err.message}`;
        }
    }

    chatFeed.addEventListener("click", (e) => {
        const chip = e.target.closest(".citation-chip");
        if (chip) {
            e.preventDefault();
            const tag = chip.dataset.tag || chip.textContent.replace(/[\[\]]/g, "").trim();
            if (tag) openCitationModal(tag);
        }
    });

    function formatMarkdownWithCitations(text) {
        let safe = escapeHtml(text);

        // Convert bold **text**
        safe = safe.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

        // Convert bullet points
        safe = safe.replace(/^[\*\-]\s+(.*)$/gm, '&bull; $1<br>');

        // Convert Citation tags [doc_id#chunk_id] into clickable interactive chips (including unicode 【】 and spaces)
        safe = safe.replace(/[\[\u3010]\s*([a-zA-Z0-9_\-]+#[a-zA-Z0-9_\-]+)\s*[\]\u3011]/g, (match, tag) => {
            return `<button type="button" class="citation-chip" data-tag="${tag}" title="Click to inspect verified source [${tag}]">[${tag}]</button>`;
        });

        // Convert newlines to paragraphs / breaks
        return safe.split("\n\n").map(p => `<p>${p.replace(/\n/g, '<br>')}</p>`).join("");
    }

    function scrollToBottom() {
        chatFeed.scrollTop = chatFeed.scrollHeight;
    }

    function escapeHtml(str) {
        if (!str) return "";
        return String(str)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
});

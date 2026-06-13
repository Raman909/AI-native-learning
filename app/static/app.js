const statusDot = document.getElementById("status-dot");
const statusLabel = document.getElementById("status-label");
const providerValue = document.getElementById("provider-value");
const vectorValue = document.getElementById("vector-value");
const geminiValue = document.getElementById("gemini-value");
const fallbackValue = document.getElementById("fallback-value");
const examplesList = document.getElementById("examples-list");
const refreshExamplesButton = document.getElementById("refresh-examples");
const askForm = document.getElementById("ask-form");
const questionInput = document.getElementById("question-input");
const maxSourcesInput = document.getElementById("max-sources");
const maxSourcesValue = document.getElementById("max-sources-value");
const submitButton = document.getElementById("submit-button");
const messageBanner = document.getElementById("message-banner");
const answerCard = document.getElementById("answer-card");
const responseProvider = document.getElementById("response-provider");
const responseTime = document.getElementById("response-time");
const sourcesCount = document.getElementById("sources-count");
const sourcesList = document.getElementById("sources-list");

function showBanner(message, mode) {
    messageBanner.textContent = message;
    messageBanner.className = `message-banner ${mode}`;
}

function hideBanner() {
    messageBanner.className = "message-banner hidden";
    messageBanner.textContent = "";
}

function setStatus(state, health = null) {
    statusDot.classList.remove("live", "error");

    if (state === "live" && health) {
        statusDot.classList.add("live");
        statusLabel.textContent = "Service healthy";
        providerValue.textContent = health.llm_status.active_provider || "--";
        vectorValue.textContent = health.vector_store || "--";
        geminiValue.textContent = health.llm_status.gemini_model || "--";
        fallbackValue.textContent = health.llm_status.hf_model || "--";
        return;
    }

    if (state === "error") {
        statusDot.classList.add("error");
        statusLabel.textContent = "Service unavailable";
    } else {
        statusLabel.textContent = "Checking service";
    }

    providerValue.textContent = "--";
    vectorValue.textContent = "--";
    geminiValue.textContent = "--";
    fallbackValue.textContent = "--";
}

function renderExamples(examples) {
    if (!examples.length) {
        examplesList.innerHTML = '<p class="muted">No examples available right now.</p>';
        return;
    }

    examplesList.innerHTML = "";
    examples.forEach((example) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "example-chip";
        button.textContent = example;
        button.addEventListener("click", () => {
            questionInput.value = example;
            questionInput.focus();
        });
        examplesList.appendChild(button);
    });
}

function renderAnswer(payload) {
    answerCard.classList.remove("empty");
    answerCard.innerHTML = "";

    const question = document.createElement("p");
    question.className = "answer-question";
    question.textContent = payload.question;

    const answer = document.createElement("div");
    answer.className = "answer-body";
    answer.textContent = payload.answer;

    answerCard.appendChild(question);
    answerCard.appendChild(answer);

    responseProvider.textContent = `Provider: ${payload.provider_used}`;
    responseTime.textContent = `Time: ${payload.processing_time_ms.toFixed(0)} ms`;
    sourcesCount.textContent = `${payload.sources.length} loaded`;

    if (!payload.sources.length) {
        sourcesList.innerHTML = '<p class="muted">No sources returned.</p>';
        return;
    }

    sourcesList.innerHTML = "";
    payload.sources.forEach((source, index) => {
        const card = document.createElement("article");
        card.className = "source-card";

        const label = document.createElement("span");
        label.textContent = `Source ${index + 1}`;

        const text = document.createElement("p");
        text.textContent = source.content;

        card.appendChild(label);
        card.appendChild(text);
        sourcesList.appendChild(card);
    });
}

function setLoading(isLoading) {
    submitButton.disabled = isLoading;
    submitButton.textContent = isLoading ? "Thinking..." : "Ask assistant";
}

async function loadHealth() {
    setStatus("loading");
    try {
        const response = await fetch("/api/v1/health");
        if (!response.ok) {
            throw new Error("Health request failed");
        }
        const payload = await response.json();
        setStatus("live", payload);
    } catch (error) {
        setStatus("error");
    }
}

async function loadExamples() {
    examplesList.innerHTML = '<p class="muted">Loading examples...</p>';
    try {
        const response = await fetch("/api/v1/examples");
        if (!response.ok) {
            throw new Error("Examples request failed");
        }
        const payload = await response.json();
        renderExamples(payload);
    } catch (error) {
        examplesList.innerHTML = '<p class="muted">Could not load examples.</p>';
    }
}

askForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    hideBanner();

    const question = questionInput.value.trim();
    const maxSources = Number(maxSourcesInput.value);

    if (question.length < 10) {
        showBanner("Please enter at least 10 characters so the assistant has enough to work with.", "error");
        return;
    }

    setLoading(true);
    showBanner("Retrieving context and generating an answer...", "info");

    try {
        const response = await fetch("/api/v1/ask", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                question,
                max_sources: maxSources,
            }),
        });

        const payload = await response.json();
        if (!response.ok) {
            throw new Error(payload.detail || "The assistant could not answer right now.");
        }

        renderAnswer(payload);
        hideBanner();
        await loadHealth();
    } catch (error) {
        showBanner(error.message || "The assistant could not answer right now.", "error");
    } finally {
        setLoading(false);
    }
});

refreshExamplesButton.addEventListener("click", loadExamples);
maxSourcesInput.addEventListener("input", () => {
    maxSourcesValue.textContent = maxSourcesInput.value;
});

loadHealth();
loadExamples();

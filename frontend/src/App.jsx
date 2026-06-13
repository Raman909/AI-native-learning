import { startTransition, useEffect, useMemo, useState } from "react";

const initialAnswer = {
  question: "",
  answer: "",
  provider_used: "",
  processing_time_ms: 0,
  sources: [],
};

function App() {
  const [question, setQuestion] = useState("");
  const [maxSources, setMaxSources] = useState(3);
  const [examples, setExamples] = useState([]);
  const [health, setHealth] = useState(null);
  const [answer, setAnswer] = useState(initialAnswer);
  const [banner, setBanner] = useState({ mode: "", message: "" });
  const [loadingExamples, setLoadingExamples] = useState(false);
  const [asking, setAsking] = useState(false);

  const sourceCountLabel = useMemo(() => {
    if (!answer.sources.length) {
      return "0 loaded";
    }
    return `${answer.sources.length} loaded`;
  }, [answer.sources]);

  useEffect(() => {
    void loadHealth();
    void loadExamples();
  }, []);

  async function loadHealth() {
    try {
      const response = await fetch("/api/v1/health");
      if (!response.ok) {
        throw new Error("Health request failed");
      }
      const payload = await response.json();
      startTransition(() => {
        setHealth(payload);
      });
    } catch (error) {
      startTransition(() => {
        setHealth({ error: true });
      });
    }
  }

  async function loadExamples() {
    setLoadingExamples(true);
    try {
      const response = await fetch("/api/v1/examples");
      if (!response.ok) {
        throw new Error("Examples request failed");
      }
      const payload = await response.json();
      startTransition(() => {
        setExamples(payload);
      });
    } catch (error) {
      startTransition(() => {
        setExamples([]);
      });
    } finally {
      setLoadingExamples(false);
    }
  }

  async function handleSubmit(event) {
    event.preventDefault();

    const trimmedQuestion = question.trim();
    if (trimmedQuestion.length < 10) {
      setBanner({
        mode: "error",
        message: "Please enter at least 10 characters so the assistant has enough context.",
      });
      return;
    }

    setAsking(true);
    setBanner({
      mode: "info",
      message: "Retrieving context and generating your answer...",
    });

    try {
      const response = await fetch("/api/v1/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: trimmedQuestion,
          max_sources: maxSources,
        }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "The assistant could not answer right now.");
      }

      startTransition(() => {
        setAnswer(payload);
        setBanner({ mode: "", message: "" });
      });
      await loadHealth();
    } catch (error) {
      setBanner({
        mode: "error",
        message: error.message || "The assistant could not answer right now.",
      });
    } finally {
      setAsking(false);
    }
  }

  const statusClass = health?.error ? "status-dot error" : health ? "status-dot live" : "status-dot";
  const statusLabel = health?.error ? "Service unavailable" : health ? "Service healthy" : "Checking service";

  return (
    <div className="page-shell">
      <div className="orb orb-one" />
      <div className="orb orb-two" />

      <header className="hero">
        <section className="hero-copy card">
          <p className="eyebrow">React frontend</p>
          <h1>Grounded Python answers, now with a dedicated React workspace.</h1>
          <p className="hero-text">
            Ask Python questions, inspect retrieved context, and monitor which model served the answer.
            The frontend talks to your FastAPI backend through the local API proxy.
          </p>
          <div className="hero-actions">
            <a className="primary-link" href="#qa-panel">Open workspace</a>
            <a className="secondary-link" href="http://127.0.0.1:8000/docs" target="_blank" rel="noreferrer">
              Backend docs
            </a>
          </div>
        </section>

        <aside className="status-card card">
          <div className="status-heading">
            <span className={statusClass} />
            <span>{statusLabel}</span>
          </div>

          <dl className="status-grid">
            <div>
              <dt>Provider</dt>
              <dd>{health?.llm_status?.active_provider || "--"}</dd>
            </div>
            <div>
              <dt>Vector store</dt>
              <dd>{health?.vector_store || "--"}</dd>
            </div>
            <div>
              <dt>Gemini model</dt>
              <dd>{health?.llm_status?.gemini_model || "--"}</dd>
            </div>
            <div>
              <dt>Fallback model</dt>
              <dd>{health?.llm_status?.hf_model || "--"}</dd>
            </div>
          </dl>
        </aside>
      </header>

      <main className="workspace">
        <section className="workspace-panel card" id="qa-panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Question workspace</p>
              <h2>Try a Python question</h2>
            </div>
            <span className="mono-chip">React + Vite</span>
          </div>

          <form className="ask-form" onSubmit={handleSubmit}>
            <label className="field-label" htmlFor="question-input">Question</label>
            <textarea
              id="question-input"
              rows={7}
              maxLength={500}
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Example: What causes RecursionError in Python and how do I fix it?"
              required
            />

            <div className="form-row">
              <label className="slider-group" htmlFor="max-sources">
                <span>Sources to show</span>
                <div className="slider-inline">
                  <input
                    id="max-sources"
                    type="range"
                    min="1"
                    max="5"
                    value={maxSources}
                    onChange={(event) => setMaxSources(Number(event.target.value))}
                  />
                  <output htmlFor="max-sources">{maxSources}</output>
                </div>
              </label>

              <button type="submit" disabled={asking}>
                {asking ? "Thinking..." : "Ask assistant"}
              </button>
            </div>
          </form>

          <section className="examples-block">
            <div className="examples-header">
              <h3>Suggested prompts</h3>
              <button className="ghost-button" type="button" onClick={() => void loadExamples()}>
                {loadingExamples ? "Refreshing..." : "Refresh"}
              </button>
            </div>

            <div className="examples-list">
              {loadingExamples && !examples.length ? <p className="muted">Loading examples...</p> : null}
              {!loadingExamples && !examples.length ? <p className="muted">No examples available right now.</p> : null}
              {examples.map((example) => (
                <button
                  key={example}
                  className="example-chip"
                  type="button"
                  onClick={() => setQuestion(example)}
                >
                  {example}
                </button>
              ))}
            </div>
          </section>
        </section>

        <section className="workspace-panel card">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Answer output</p>
              <h2>Response</h2>
            </div>
            <div className="response-meta">
              <span className="mono-chip">Provider: {answer.provider_used || "--"}</span>
              <span className="mono-chip">
                Time: {answer.processing_time_ms ? `${Math.round(answer.processing_time_ms)} ms` : "--"}
              </span>
            </div>
          </div>

          {banner.message ? <div className={`message-banner ${banner.mode}`}>{banner.message}</div> : null}

          <article className={`answer-card ${answer.answer ? "" : "empty"}`}>
            {answer.answer ? (
              <>
                <p className="answer-question">{answer.question}</p>
                <div className="answer-body">{answer.answer}</div>
              </>
            ) : (
              <p className="empty-state">
                Your grounded answer will appear here, along with the retrieved source snippets.
              </p>
            )}
          </article>

          <section className="sources-section">
            <div className="sources-header">
              <h3>Retrieved sources</h3>
              <span className="muted">{sourceCountLabel}</span>
            </div>

            <div className="sources-list">
              {!answer.sources.length ? <p className="muted">No sources yet.</p> : null}
              {answer.sources.map((source, index) => (
                <article className="source-card" key={`${index}-${source.content.slice(0, 24)}`}>
                  <span>Source {index + 1}</span>
                  <p>{source.content}</p>
                </article>
              ))}
            </div>
          </section>
        </section>
      </main>
    </div>
  );
}

export default App;

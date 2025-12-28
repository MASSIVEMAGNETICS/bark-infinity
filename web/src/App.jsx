import { useEffect, useMemo, useState } from 'react';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';

const clamp = (value, min, max) => Math.min(Math.max(value, min), max);

const formatSeconds = (seconds) => {
  if (!seconds || Number.isNaN(seconds)) {
    return '—';
  }
  const wholeSeconds = Math.floor(seconds);
  const mins = Math.floor(wholeSeconds / 60);
  const secs = clamp(wholeSeconds % 60, 0, 59);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
};

const defaultPrompt = `Welcome to Bark Infinity React!\n\n` +
  `Try typing any text prompt and press Generate to create an audio clip.\n` +
  `Adjust temperature sliders for exploration or keep them steady for more deterministic results.`;

export default function App() {
  const [text, setText] = useState(defaultPrompt);
  const [prompts, setPrompts] = useState([]);
  const [selectedPrompt, setSelectedPrompt] = useState('');
  const [textTemp, setTextTemp] = useState(0.7);
  const [waveformTemp, setWaveformTemp] = useState(0.7);
  const [seed, setSeed] = useState('');
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('');
  const [error, setError] = useState('');
  const [audioUrl, setAudioUrl] = useState('');
  const [audioMeta, setAudioMeta] = useState(null);

  const hasPrompts = prompts.length > 0;

  const sortedPrompts = useMemo(
    () => prompts.slice().sort((a, b) => a.name.localeCompare(b.name)),
    [prompts]
  );

  const loadPrompts = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/prompts`);
      if (!response.ok) {
        throw new Error('Failed to load voices');
      }
      const data = await response.json();
      setPrompts(data);
    } catch (fetchError) {
      console.error(fetchError);
      setStatus('');
      setError('Unable to load saved voices. The backend may not be running yet.');
    }
  };

  useEffect(() => {
    loadPrompts();
  }, []);

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!text.trim()) {
      setError('Please provide some text to generate audio.');
      return;
    }

    setLoading(true);
    setStatus('Generating audio – this may take a little while for long prompts.');
    setError('');
    setAudioUrl('');
    setAudioMeta(null);

    const payload = {
      text,
      history_prompt: selectedPrompt || null,
      text_temp: Number(textTemp),
      waveform_temp: Number(waveformTemp),
      seed: seed !== '' ? Number(seed) : null
    };

    try {
      const response = await fetch(`${API_BASE_URL}/api/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        throw new Error(detail?.detail ?? 'Generation failed');
      }

      const data = await response.json();
      const audioDataUrl = `data:audio/wav;base64,${data.audio_base64}`;
      setAudioUrl(audioDataUrl);
      setAudioMeta(data);
      setStatus('Audio ready! You can play it below or download the clip.');
    } catch (submitError) {
      console.error(submitError);
      setError(submitError.message ?? 'Unexpected error during generation.');
      setStatus('');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setText(defaultPrompt);
    setSelectedPrompt('');
    setTextTemp(0.7);
    setWaveformTemp(0.7);
    setSeed('');
    setAudioUrl('');
    setAudioMeta(null);
    setStatus('');
    setError('');
  };

  const handleSeedChange = (value) => {
    if (value === '') {
      setSeed('');
      return;
    }
    const asNumber = Number(value);
    if (!Number.isNaN(asNumber)) {
      setSeed(Math.max(0, Math.floor(asNumber)).toString());
    }
  };

  return (
    <div className="app-wrapper">
      <header className="app-header">
        <h1>Bark Infinity React</h1>
        <p>Generate Bark audio from a modern React interface that talks to the Bark Infinity API.</p>
      </header>

      <main className="card">
        <form className="form-grid" onSubmit={handleSubmit}>
          <div className="field-group">
            <label htmlFor="text-input">Text prompt</label>
            <textarea
              id="text-input"
              value={text}
              onChange={(event) => setText(event.target.value)}
              placeholder="Describe the scene, dialogue, or narration you would like Bark to read."
            />
          </div>

          <div className="field-group">
            <label htmlFor="prompt-select">Voice / history prompt</label>
            <select
              id="prompt-select"
              value={selectedPrompt}
              onChange={(event) => setSelectedPrompt(event.target.value)}
            >
              <option value="">Default Bark voice</option>
              {sortedPrompts.map((prompt) => (
                <option key={prompt.path} value={prompt.path}>
                  {prompt.name}
                </option>
              ))}
            </select>
            {!hasPrompts && (
              <small className="empty-state">
                No saved voices found yet. Generate audio in Bark Infinity to create speaker files.
              </small>
            )}
          </div>

          <div className="slider-row">
            <div className="slider">
              <label htmlFor="text-temp">Text temperature: {Number(textTemp).toFixed(2)}</label>
              <input
                id="text-temp"
                type="range"
                min="0"
                max="1.5"
                step="0.05"
                value={textTemp}
                onChange={(event) => setTextTemp(event.target.value)}
              />
            </div>

            <div className="slider">
              <label htmlFor="waveform-temp">Waveform temperature: {Number(waveformTemp).toFixed(2)}</label>
              <input
                id="waveform-temp"
                type="range"
                min="0"
                max="1.5"
                step="0.05"
                value={waveformTemp}
                onChange={(event) => setWaveformTemp(event.target.value)}
              />
            </div>
          </div>

          <div className="field-group">
            <label htmlFor="seed-input">Seed (optional)</label>
            <input
              id="seed-input"
              type="number"
              min="0"
              value={seed}
              onChange={(event) => handleSeedChange(event.target.value)}
              placeholder="Leave blank for random"
            />
          </div>

          {status && <div className="status-banner">{status}</div>}
          {error && <div className="status-banner error">{error}</div>}

          <div className="button-row">
            <button className="primary" type="submit" disabled={loading}>
              {loading ? 'Generating…' : 'Generate audio'}
            </button>
            <button className="secondary" type="button" onClick={handleReset} disabled={loading}>
              Reset form
            </button>
            <button className="secondary" type="button" onClick={loadPrompts} disabled={loading}>
              Refresh voices
            </button>
          </div>
        </form>

        {audioUrl ? (
          <section className="audio-card">
            <div>
              <h3>Latest render</h3>
              <p>
                Sample rate: {audioMeta?.sample_rate ?? '—'} Hz · Duration: {formatSeconds(audioMeta?.duration_seconds)}
                {audioMeta?.history_prompt_name ? ` · Voice: ${audioMeta.history_prompt_name}` : ''}
              </p>
            </div>

            <div className="audio-player">
              <audio controls src={audioUrl} />
              <div className="button-row">
                <a className="primary" download={audioMeta?.filename ?? 'bark-generation.wav'} href={audioUrl}>
                  Download clip
                </a>
                <button
                  className="secondary"
                  type="button"
                  onClick={() => navigator.clipboard?.writeText(audioMeta?.history_prompt_path ?? '')}
                  disabled={!audioMeta?.history_prompt_path}
                >
                  Copy voice path
                </button>
              </div>
            </div>
          </section>
        ) : (
          <div className="empty-state">
            <p>Audio results will appear here after you generate a clip.</p>
          </div>
        )}
      </main>
    </div>
  );
}

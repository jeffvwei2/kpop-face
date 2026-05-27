import { useState, useCallback } from 'react'
import './App.css'

export default function App() {
  const [preview, setPreview]   = useState(null)
  const [results, setResults]   = useState(null)
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState(null)
  const [dragging, setDragging] = useState(false)

  const runPrediction = useCallback(async (file) => {
    if (!file || !file.type.startsWith('image/')) {
      setError('Please drop an image file.')
      return
    }
    setPreview(URL.createObjectURL(file))
    setResults(null)
    setError(null)
    setLoading(true)

    const form = new FormData()
    form.append('image', file)

    try {
      const res  = await fetch('/predict', { method: 'POST', body: form })
      const data = await res.json()
      if (data.error) throw new Error(data.error)
      setResults(data.predictions)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) runPrediction(file)
  }, [runPrediction])

  const reset = () => { setPreview(null); setResults(null); setError(null) }
  const top   = results?.[0]

  return (
    <div className="app">
      <h1>KPop Face ID</h1>

      <label
        className={`dropzone ${dragging ? 'dragging' : ''} ${preview ? 'has-image' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <input type="file" accept="image/*" onChange={(e) => e.target.files[0] && runPrediction(e.target.files[0])} hidden />
        {preview
          ? <img src={preview} alt="preview" className="preview-img" />
          : <div className="drop-hint">
              <div className="drop-icon">🖼️</div>
              <p>Drop a photo here or <span className="link">click to upload</span></p>
            </div>
        }
      </label>

      {loading && <p className="status">Analyzing…</p>}
      {error   && <p className="status error">{error}</p>}

      {results && (
        <div className="results">
          <div className="top-result">
            <span className="top-name">{top.name}</span>
            <span className="top-conf">{top.confidence}%</span>
          </div>
          <div className="runners-up">
            {results.slice(1).map((r) => (
              <div key={r.name} className="runner-row">
                <span className="runner-name">{r.name}</span>
                <div className="bar-track">
                  <div className="bar-fill" style={{ width: `${(r.confidence / top.confidence) * 100}%` }} />
                </div>
                <span className="runner-conf">{r.confidence}%</span>
              </div>
            ))}
          </div>
          <button className="reset-btn" onClick={reset}>Try another</button>
        </div>
      )}
    </div>
  )
}

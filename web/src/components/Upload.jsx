import { useState, useCallback, useRef, useEffect } from 'react'
import './Upload.css'
import { Dropzone } from './Dropzone'
import { Results } from './Results'

export default function Upload({ item, onReset }) {
  const [preview, setPreview]   = useState(null)
  const [results, setResults] = useState(null)
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState(null)
  const [dragging, setDragging] = useState(false)
  const previewUrlRef = useRef(null)

  
  // object URLs aren't garbage-collected on their own; revoke the old one
  // whenever it's replaced, and on unmount so it doesn't leak for the tab's lifetime.
  const setPreviewFile = useCallback((file) => {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current)
    previewUrlRef.current = file ? URL.createObjectURL(file) : null
    setPreview(previewUrlRef.current)
  }, [])

  useEffect(() => () => {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current)
  }, [])

  // while viewing a history item, its preview/results take over the display
  // instead of being copied into local state
  const displayPreview = item ? item.preview : preview
  const displayResults = item ? item.results : results

  const runPrediction = useCallback(async (file) => {
    if (!file || !file.type.startsWith('image/')) {
      setError('Please drop an image file.')
      return
    }
    setPreviewFile(file)
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
  }, [setPreviewFile])

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) runPrediction(file)
  }, [runPrediction])

  const reset = () => {
    const wasLive = !item
    onReset(wasLive ? { preview, results } : null)

    if (wasLive) {
      // ownership of this blob URL now belongs to the history entry — don't revoke it
      previewUrlRef.current = null
      setPreview(null)
    } else {
      setPreviewFile(null)
    }
    setResults(null);
    setError(null)
  }

  return (
    <div>
      <h1>KPop Face ID</h1>
      <Dropzone
        preview={displayPreview}
        dragging={dragging}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onFileChange={(e) => e.target.files[0] && runPrediction(e.target.files[0])}
      />

      {!item && loading && <p className="status">Analyzing…</p>}
      {!item && error   && <p className="status error">{error}</p>}

      {displayResults && (
        <>
          <Results results={displayResults} />
          <button className="reset-btn" onClick={reset}>Try another</button>
        </>
      )}
    </div>
  )
}

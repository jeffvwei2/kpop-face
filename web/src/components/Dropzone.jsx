import './Dropzone.css'
import Preview from './Preview'
export function Dropzone({ preview, dragging, onDragOver, onDragLeave, onDrop, onFileChange }) {
  return (
    <label
      className={`dropzone ${dragging ? 'dragging' : ''} ${preview ? 'has-image' : ''}`}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
    >
      <input type="file" accept="image/*" onChange={onFileChange} hidden />
      {/* {preview
        ? <img src={preview} alt="preview" className="preview-img" />
        : <div className="drop-hint">
            <div className="drop-icon">🖼️</div>
            <p>Drop a photo here or <span className="link">click to upload</span></p>
          </div>
      } */}
    <Preview preview={preview} />
    </label>
  )
}

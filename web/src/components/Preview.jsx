export default function Preview({preview}) {
  return (<>
    {preview
        ? <img src={preview} alt="preview" className="preview-img" />
        : <div className="drop-hint">
            <div className="drop-icon">🖼️</div>
            <p>Drop a photo here or <span className="link">click to upload</span></p>
          </div>
      }
    </>)
}
import './Results.css'

export function Results({ results }) {
  const top = results[0]
  return (
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
      
    </div>
  )
}

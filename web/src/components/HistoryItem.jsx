import './HistoryItem.css'
import Preview from "./Preview"

export default function HistoryItem({ item, onItemSelect, idx, isSelected }) {
  const top = item?.results?.[0]
  return (
    <div
      className={`history-item ${isSelected ? 'selected' : ''}`}
      onClick={() => onItemSelect(idx)}
    >
      <div className="history-item-preview">
        <Preview preview={item.preview}/>
      </div>
      <div className="history-item-info">
        <p className="history-item-name">{top?.name}</p>
        <p className="history-item-conf">{top?.confidence}%</p>
      </div>
    </div>
  )
}
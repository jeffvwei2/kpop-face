import "./History.css"
import HistoryItem from "./HistoryItem"

export default function History({ history, onItemSelect, currentItem }) {

  return (
    <div className="history">
      <h2 className="history-title">History</h2>
      <div className="history-list">
        {history?.length && history.map((item, idx) => (
          <HistoryItem
            key={idx}
            item={item}
            idx={idx}
            isSelected={item === currentItem}
            onItemSelect={onItemSelect}
          />
        ))}
      </div>
    </div>
  )
}
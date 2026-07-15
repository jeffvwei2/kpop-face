import { useState } from 'react'
import './App.css'
import Upload from './components/Upload'
import Sidebar from './components/Sidebar'
import History from './components/History'

export default function App() {
  const [history, setHistory] = useState(null)
  const [currentItem, setCurrentItem] = useState(null)


  function handleReset(completedItem) {
    if (!completedItem) {
      setCurrentItem(null)
      return
    }

    setHistory(prev => {
      if (!prev) return [completedItem]
      if (prev.length >= 5) {
        // evicting the oldest entry — release its blob URL so it doesn't leak
        URL.revokeObjectURL(prev[prev.length - 1].preview)
        return [completedItem, ...prev.slice(0, -1)]
      }
      return [completedItem, ...prev]
    })
    setCurrentItem(null)
  }

  function handleItemSelect(idx) {
    setCurrentItem(history[idx])
  }


  return (
    <div className="app">
      <Upload item={currentItem} onReset={handleReset}/>
      <Sidebar>
        <History history={history} onItemSelect={handleItemSelect} currentItem={currentItem}/>
      </Sidebar>
    </div>
  )
}

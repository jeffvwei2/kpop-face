import { useState } from 'react'
import './Sidebar.css'

export default function Sidebar({ children }) {
  const [open, setOpen] = useState(false)

  return (
    <>
      <button
        className={`sidebar-toggle ${open ? 'open' : ''}`}
        onClick={() => setOpen((o) => !o)}
        aria-label={open ? 'Close history' : 'Open history'}
        aria-expanded={open}
      >
        {open ? '‹' : '›'}
      </button>

      {open && <div className="sidebar-backdrop" onClick={() => setOpen(false)} />}

      <aside className={`sidebar ${open ? 'open' : ''}`}>
        {children}
      </aside>
    </>
  )
}

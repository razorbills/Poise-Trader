import React from 'react'

export default function Terminal({ alerts }) {
  return (
    <div className="terminal">
      {(alerts || []).map((a, idx) => (
        <div key={idx} className={`terminal-line ${a.severity || 'info'}`}>
          <span className="ts">[{a.timestamp}]</span>
          <span className="cat"> {String(a.category || '').toUpperCase()}</span>
          <span className="msg"> {a.message}</span>
        </div>
      ))}
      {(!alerts || alerts.length === 0) && (
        <div className="terminal-line info">No alerts yet</div>
      )}
    </div>
  )
}

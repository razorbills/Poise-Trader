import React, { useMemo, useState } from 'react'
import socket from '../socket'

export default function Positions({ portfolio }) {
  const positions = useMemo(() => {
    const p = portfolio?.positions || {}
    return Object.entries(p).map(([symbol, pos]) => ({ symbol, ...pos }))
  }, [portfolio])

  const [closeAmt, setCloseAmt] = useState({})
  const [status, setStatus] = useState('')

  const send = (action, payload) => new Promise((resolve) => {
    const onAck = (msg) => { if (msg?.action === action) { cleanup(); resolve({ ok: true }) } }
    const onErr = (msg) => { if (msg?.action === action) { cleanup(); resolve({ ok: false, error: msg.error }) } }
    const cleanup = () => { socket.off('control_ack', onAck); socket.off('control_error', onErr) }
    socket.on('control_ack', onAck)
    socket.on('control_error', onErr)
    socket.emit('control', { action, payload })
  })

  const closePosition = async (symbol, amountUsd) => {
    setStatus('Closing...')
    const res = await send('close_position', { symbol, amount_usd: amountUsd })
    setStatus(res.ok ? 'Close requested' : `Error: ${res.error}`)
    setTimeout(() => setStatus(''), 2000)
  }

  return (
    <div className="positions">
      <table className="table">
        <thead>
          <tr>
            <th>Symbol</th>
            <th>Qty</th>
            <th>Price</th>
            <th>Value</th>
            <th>Cost</th>
            <th>Unrealized PnL</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {positions.length === 0 && (
            <tr><td colSpan="7" style={{ opacity: 0.7 }}>No open positions</td></tr>
          )}
          {positions.map((pos) => (
            <tr key={pos.symbol}>
              <td>{pos.symbol}</td>
              <td>{Number(pos.quantity).toFixed(6)}</td>
              <td>${Number(pos.current_price).toFixed(2)}</td>
              <td>${Number(pos.current_value).toFixed(2)}</td>
              <td>${Number(pos.cost_basis).toFixed(2)}</td>
              <td style={{ color: pos.unrealized_pnl >= 0 ? '#4CAF50' : '#f44336' }}>
                ${Number(pos.unrealized_pnl).toFixed(2)}
              </td>
              <td>
                <div className="actions">
                  <button className="btn" onClick={() => closePosition(pos.symbol, pos.current_value)}>Close Full</button>
                  <input className="input" type="number" min="0" step="0.1" placeholder="$ amount"
                         value={closeAmt[pos.symbol] || ''}
                         onChange={(e) => setCloseAmt({ ...closeAmt, [pos.symbol]: e.target.value })} />
                  <button className="btn" onClick={() => closePosition(pos.symbol, Number(closeAmt[pos.symbol] || 0))}>Close Partial</button>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="hint">{status}</div>
    </div>
  )
}

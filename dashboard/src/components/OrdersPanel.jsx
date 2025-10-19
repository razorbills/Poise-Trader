import React, { useState } from 'react'
import socket from '../socket'

export default function OrdersPanel() {
  const [symbol, setSymbol] = useState('BTC/USDT')
  const [side, setSide] = useState('buy')
  const [amount, setAmount] = useState(10)
  const [tp, setTp] = useState('')
  const [sl, setSl] = useState('')
  const [status, setStatus] = useState('')

  const send = (action, payload) => new Promise((resolve) => {
    const onAck = (msg) => { if (msg?.action === action) { cleanup(); resolve({ ok: true }) } }
    const onErr = (msg) => { if (msg?.action === action) { cleanup(); resolve({ ok: false, error: msg.error }) } }
    const cleanup = () => { socket.off('control_ack', onAck); socket.off('control_error', onErr) }
    socket.on('control_ack', onAck)
    socket.on('control_error', onErr)
    socket.emit('control', { action, payload })
  })

  const placeOrder = async () => {
    setStatus('Sending...')
    const res = await send('place_order', {
      symbol,
      side,
      amount: Number(amount),
      take_profit: tp === '' ? undefined : Number(tp),
      stop_loss: sl === '' ? undefined : Number(sl)
    })
    setStatus(res.ok ? 'Order sent' : `Error: ${res.error}`)
    setTimeout(() => setStatus(''), 2000)
  }

  return (
    <div className="orders-panel">
      <div className="field">
        <label>Symbol</label>
        <input value={symbol} onChange={e => setSymbol(e.target.value)} />
      </div>
      <div className="field">
        <label>Side</label>
        <select value={side} onChange={e => setSide(e.target.value)}>
          <option value="buy">Buy</option>
          <option value="sell">Sell</option>
        </select>
      </div>
      <div className="field">
        <label>Amount ($)</label>
        <input type="number" min="0" step="0.1" value={amount} onChange={e => setAmount(e.target.value)} />
      </div>
      <div className="field">
        <label>Take Profit ($)</label>
        <input type="number" min="0" step="0.0001" placeholder="optional" value={tp} onChange={e => setTp(e.target.value)} />
      </div>
      <div className="field">
        <label>Stop Loss ($)</label>
        <input type="number" min="0" step="0.0001" placeholder="optional" value={sl} onChange={e => setSl(e.target.value)} />
      </div>
      <button className="btn" onClick={placeOrder}>Place Order</button>
      <div className="hint">{status}</div>
    </div>
  )
}

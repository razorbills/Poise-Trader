import React, { useEffect, useState } from 'react'
import socket from '../socket'

export default function ControlPanel() {
  const [status, setStatus] = useState({})

  const [running, setRunning] = useState(false)
  const [mode, setMode] = useState('PRECISION')
  const [symbols, setSymbols] = useState('BTC/USDT,ETH/USDT,SOL/USDT')
  const [minConfidence, setMinConfidence] = useState(0.15)
  const [riskMultiplier, setRiskMultiplier] = useState('')
  const [minTradeSize, setMinTradeSize] = useState(1.0)
  const [cycleSleep, setCycleSleep] = useState('')

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const response = await fetch('http://localhost:5000/api/status')
        if (response.ok) {
          const s = await response.json()
          setStatus(s)
          if (typeof s.running === 'boolean') setRunning(s.running)
          if (s.mode) setMode(s.mode)
          if (Array.isArray(s.symbols)) setSymbols(s.symbols.join(','))
          if (typeof s.min_confidence === 'number') setMinConfidence(s.min_confidence)
          if (typeof s.risk_multiplier === 'number') setRiskMultiplier(s.risk_multiplier)
          if (typeof s.min_trade_size === 'number') setMinTradeSize(s.min_trade_size)
          if (typeof s.cycle_sleep === 'number') setCycleSleep(s.cycle_sleep)
        }
      } catch (error) {
        console.error('Failed to fetch status:', error)
      }
    }
    
    fetchStatus()
    const interval = setInterval(fetchStatus, 5000)
    return () => clearInterval(interval)
  }, [])

  // Control helpers
  const send = (action, payload) => {
    return new Promise((resolve) => {
      const onAck = (msg) => {
        if (msg?.action === action) {
          socket.off('control_ack', onAck)
          socket.off('control_error', onErr)
          resolve({ ok: true })
        }
      }
      const onErr = (msg) => {
        if (msg?.action === action) {
          socket.off('control_ack', onAck)
          socket.off('control_error', onErr)
          resolve({ ok: false, error: msg.error })
        }
      }
      socket.on('control_ack', onAck)
      socket.on('control_error', onErr)
      socket.emit('control', { action, payload })
    })
  }

  const applySymbols = async () => {
    const list = symbols.split(',').map(s => s.trim()).filter(Boolean)
    await send('set_symbols', { symbols: list })
  }

  const applyRisk = async () => {
    await send('set_risk', {
      min_confidence: Number(minConfidence),
      risk_multiplier: riskMultiplier === '' ? undefined : Number(riskMultiplier),
      min_trade_size: Number(minTradeSize)
    })
  }

  const applyMode = async () => {
    try {
      // Try WebSocket first
      const wsResult = await send('set_mode', { mode })
      if (!wsResult.ok) {
        // Fallback to REST API
        const response = await fetch('http://localhost:5000/api/mode', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ mode })
        })
        if (!response.ok) {
          console.error('Failed to set mode')
        }
      }
    } catch (error) {
      console.error('Failed to apply mode:', error)
    }
  }

  const toggleTrading = async () => {
    const next = !running
    try {
      // Try WebSocket first
      const wsResult = await send('toggle_trading', { running: next })
      if (!wsResult.ok) {
        // Fallback to REST API
        const endpoint = next ? '/api/start' : '/api/stop'
        const response = await fetch(`http://localhost:5000${endpoint}`, { 
          method: 'POST',
          headers: { 'Content-Type': 'application/json' }
        })
        if (response.ok) {
          setRunning(next)
        }
      } else {
        setRunning(next)
      }
    } catch (error) {
      console.error('Failed to toggle trading:', error)
    }
  }

  const applyCycle = async () => {
    await send('set_cycle_interval', { seconds: cycleSleep === '' ? null : Number(cycleSleep) })
  }

  return (
    <div className="control-panel">
      <div className="control-row">
        <button className={`btn ${running ? 'btn-stop' : 'btn-start'}`} onClick={toggleTrading}>
          {running ? 'Pause Trading' : 'Start Trading'}
        </button>
        <div className="field">
          <label>Mode</label>
          <select value={mode} onChange={e => setMode(e.target.value)}>
            <option value="AGGRESSIVE">AGGRESSIVE</option>
            <option value="PRECISION">PRECISION</option>
          </select>
          <button className="btn" onClick={applyMode}>Apply</button>
        </div>
        <div className="field">
          <label>Cycle (s)</label>
          <input type="number" min="0" step="1" placeholder="auto" value={cycleSleep}
                 onChange={e => setCycleSleep(e.target.value)} />
          <button className="btn" onClick={applyCycle}>Apply</button>
        </div>
      </div>

      <div className="control-row">
        <div className="field field-wide">
          <label>Symbols (comma-separated)</label>
          <input type="text" value={symbols} onChange={e => setSymbols(e.target.value)} />
          <button className="btn" onClick={applySymbols}>Apply</button>
        </div>
      </div>

      <div className="control-row">
        <div className="field">
          <label>Min Confidence</label>
          <input type="number" min="0" max="1" step="0.01" value={minConfidence}
                 onChange={e => setMinConfidence(e.target.value)} />
        </div>
        <div className="field">
          <label>Risk Multiplier</label>
          <input type="number" step="0.1" placeholder="leave blank" value={riskMultiplier}
                 onChange={e => setRiskMultiplier(e.target.value)} />
        </div>
        <div className="field">
          <label>Min Trade Size ($)</label>
          <input type="number" min="0" step="0.1" value={minTradeSize}
                 onChange={e => setMinTradeSize(e.target.value)} />
        </div>
        <button className="btn" onClick={applyRisk}>Apply Risk</button>
      </div>

      <div className="status-row">
        <div>Status: <b style={{color: running ? '#4CAF50' : '#f44336'}}>{running ? 'RUNNING' : 'STOPPED'}</b></div>
        <div>Mode: <b>{mode}</b></div>
        <div>Positions: <b>{status.positions ?? 0}</b></div>
        <div>Win Rate: <b>{(status.win_rate ?? 0).toFixed(1)}%</b></div>
      </div>
    </div>
  )
}

import React, { useEffect, useState, useMemo } from 'react'
import socket from './socket'
import PnLChart from './components/PnLChart.jsx'
import RiskHeatmap from './components/RiskHeatmap.jsx'
import Terminal from './components/Terminal.jsx'
import ControlPanel from './components/ControlPanel.jsx'
import OrdersPanel from './components/OrdersPanel.jsx'
import Positions from './components/Positions.jsx'

export default function App() {
  const [metrics, setMetrics] = useState({ total_pnl: 0, win_rate: 0, active_positions: 0, daily_volume: 0 })
  const [pnlHistory, setPnlHistory] = useState([])
  const [alerts, setAlerts] = useState([])
  const [portfolio, setPortfolio] = useState({ total_value: 0, cash: 0, positions: {} })
  const [connected, setConnected] = useState(false)
  const [botConnected, setBotConnected] = useState(false)

  useEffect(() => {
    const onMetrics = (data) => {
      console.log('Received metrics:', data)
      setMetrics({
        total_pnl: data.total_pnl ?? 0,
        win_rate: data.win_rate ?? 0,
        active_positions: data.active_positions ?? 0,
        daily_volume: data.daily_volume ?? 0,
      })
      setPnlHistory(data.pnl_history ?? [])
    }

    const onAlert = (alert) => {
      setAlerts((prev) => [alert, ...prev].slice(0, 50))
    }

    const onConnect = () => {
      setConnected(true)
      console.log('Dashboard connected to backend')
    }

    const onDisconnect = () => {
      setConnected(false)
      console.log('Dashboard disconnected from backend')
    }

    socket.on('connect', onConnect)
    socket.on('disconnect', onDisconnect)
    socket.on('metrics_update', onMetrics)
    socket.on('new_alert', onAlert)

    // Fetch initial metrics with proper error handling
    const fetchMetrics = async () => {
      try {
        const response = await fetch('http://localhost:5000/api/metrics')
        if (response.ok) {
          const data = await response.json()
          onMetrics(data)
        }
      } catch (error) {
        console.error('Failed to fetch metrics:', error)
      }
    }

    // Fetch portfolio with proper error handling
    const fetchPortfolio = async () => {
      try {
        const response = await fetch('http://localhost:5000/api/portfolio')
        if (response.ok) {
          const data = await response.json()
          setPortfolio(data)
        }
      } catch (error) {
        console.error('Failed to fetch portfolio:', error)
      }
    }
    
    // Check bot connection status
    const checkBotConnection = async () => {
      try {
        const response = await fetch('http://localhost:5000/api/status')
        if (response.ok) {
          const data = await response.json()
          setBotConnected(data.connected || false)
        }
      } catch (error) {
        setBotConnected(false)
      }
    }

    // Initial fetch
    fetchMetrics()
    fetchPortfolio()
    checkBotConnection()

    // Poll for updates
    const metricsInterval = setInterval(fetchMetrics, 5000)
    const portfolioInterval = setInterval(fetchPortfolio, 3000)
    const statusInterval = setInterval(checkBotConnection, 2000)

    return () => {
      socket.off('connect', onConnect)
      socket.off('disconnect', onDisconnect)
      socket.off('metrics_update', onMetrics)
      socket.off('new_alert', onAlert)
      clearInterval(metricsInterval)
      clearInterval(portfolioInterval)
      clearInterval(statusInterval)
    }
  }, [])

  const pnlSeries = useMemo(() => (pnlHistory || []).map((p) => ({
    ts: p.timestamp,
    pnl: p.value
  })), [pnlHistory])

  return (
    <div className="page">
      <header className="header">
        <h1>Poise Trader – Institutional Dashboard</h1>
        <div className="connection-status" style={{ fontSize: '12px', marginTop: '5px', display: 'flex', gap: '15px', justifyContent: 'center' }}>
          <span>Backend: {connected ? '🟢 Connected' : '🔴 Disconnected'}</span>
          <span>Bot: {botConnected ? '🟢 Real Data' : '🟡 Waiting for bot'}</span>
        </div>
        <div className="kpis">
          <div className="kpi">
            <div className="kpi-title">Total P&L</div>
            <div className="kpi-value" style={{ color: metrics.total_pnl >= 0 ? '#4CAF50' : '#f44336' }}>
              ${metrics.total_pnl >= 0 ? '+' : ''}{metrics.total_pnl.toFixed(2)}
            </div>
          </div>
          <div className="kpi">
            <div className="kpi-title">Win Rate</div>
            <div className="kpi-value">{(metrics.win_rate * 100).toFixed(1)}%</div>
          </div>
          <div className="kpi">
            <div className="kpi-title">Active Positions</div>
            <div className="kpi-value">{metrics.active_positions}</div>
          </div>
          <div className="kpi">
            <div className="kpi-title">Daily Volume</div>
            <div className="kpi-value">${metrics.daily_volume?.toLocaleString?.() ?? metrics.daily_volume}</div>
          </div>
        </div>
      </header>

      <main className="layout">
        <section className="panel controls">
          <div className="panel-title">Controls</div>
          <ControlPanel />
        </section>

        <section className="panel pnl">
          <div className="panel-title">Real-Time P&L</div>
          <PnLChart data={pnlSeries} />
        </section>

        <section className="panel heatmap">
          <div className="panel-title">Risk Heat Map</div>
          <RiskHeatmap data={pnlSeries} />
        </section>

        <section className="panel orders">
          <div className="panel-title">Orders</div>
          <OrdersPanel />
        </section>

        <section className="panel positions">
          <div className="panel-title">Positions</div>
          <Positions portfolio={portfolio || { total_value: 5000, cash: 5000, positions: {} }} />
        </section>

        <section className="panel terminal">
          <div className="panel-title">Bloomberg-style Terminal (Alerts)</div>
          <Terminal alerts={alerts} />
        </section>
      </main>
    </div>
  )
}

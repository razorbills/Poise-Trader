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

  useEffect(() => {
    const onMetrics = (data) => {
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

    socket.on('metrics_update', onMetrics)
    socket.on('new_alert', onAlert)

    // Fetch initial metrics
    fetch('/api/metrics').then(r => r.json()).then(d => onMetrics(d)).catch(() => {})

    // Poll portfolio
    const interval = setInterval(() => {
      fetch('/api/portfolio')
        .then(r => r.json())
        .then(data => setPortfolio(data))
        .catch(() => {})
    }, 3000)

    return () => {
      socket.off('metrics_update', onMetrics)
      socket.off('new_alert', onAlert)
      clearInterval(interval)
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
        <div className="kpis">
          <div className="kpi">
            <div className="kpi-title">Total P&L</div>
            <div className="kpi-value">${metrics.total_pnl.toFixed(2)}</div>
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
          <Positions portfolio={portfolio} />
        </section>

        <section className="panel terminal">
          <div className="panel-title">Bloomberg-style Terminal (Alerts)</div>
          <Terminal alerts={alerts} />
        </section>
      </main>
    </div>
  )
}

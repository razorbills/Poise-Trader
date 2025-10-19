import React from 'react'
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from 'recharts'

export default function PnLChart({ data }) {
  const formatted = (data || []).map(d => ({
    ts: new Date(d.ts).toLocaleTimeString(),
    pnl: Number(d.pnl) || 0
  }))

  return (
    <div className="chart-container">
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={formatted} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2d3561" />
          <XAxis dataKey="ts" stroke="#8fa1ff" />
          <YAxis stroke="#8fa1ff" />
          <Tooltip contentStyle={{ background: '#1a1d3a', border: '1px solid #2d3561' }} />
          <Line type="monotone" dataKey="pnl" stroke="#4CAF50" dot={false} strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

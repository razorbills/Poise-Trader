import React, { useMemo } from 'react'
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, ReferenceLine } from 'recharts'

export default function PnLChart({ data }) {
  const formatted = useMemo(() => {
    if (!data || data.length === 0) {
      // Start with initial data point at 0
      return [{ ts: new Date().toLocaleTimeString(), pnl: 0, display: '$0.00' }]
    }
    
    return data.map(d => ({
      ts: d.timestamp ? new Date(d.timestamp).toLocaleTimeString() : new Date().toLocaleTimeString(),
      pnl: typeof d.value === 'number' ? d.value : (typeof d.pnl === 'number' ? d.pnl : 0),
      display: `$${(typeof d.value === 'number' ? d.value : (typeof d.pnl === 'number' ? d.pnl : 0)).toFixed(2)}`
    }))
  }, [data])

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload[0]) {
      const value = payload[0].value
      return (
        <div style={{ background: '#1a1d3a', padding: '10px', border: '1px solid #2d3561', borderRadius: '5px' }}>
          <p style={{ color: '#8fa1ff', margin: 0 }}>{payload[0].payload.ts}</p>
          <p style={{ color: value >= 0 ? '#4CAF50' : '#f44336', margin: 0, fontWeight: 'bold' }}>
            P&L: ${value >= 0 ? '+' : ''}{value.toFixed(2)}
          </p>
        </div>
      )
    }
    return null
  }

  const lineColor = formatted.length > 0 && formatted[formatted.length - 1].pnl >= 0 ? '#4CAF50' : '#f44336'

  return (
    <div className="chart-container">
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={formatted} margin={{ top: 5, right: 20, bottom: 5, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2d3561" />
          <XAxis 
            dataKey="ts" 
            stroke="#8fa1ff"
            tick={{ fontSize: 12 }}
          />
          <YAxis 
            stroke="#8fa1ff"
            tick={{ fontSize: 12 }}
            domain={['dataMin - 10', 'dataMax + 10']}
            tickFormatter={(value) => `$${value.toFixed(0)}`}
          />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine y={0} stroke="#666" strokeDasharray="5 5" />
          <Line 
            type="monotone" 
            dataKey="pnl" 
            stroke={lineColor} 
            dot={false} 
            strokeWidth={2}
            animationDuration={500}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

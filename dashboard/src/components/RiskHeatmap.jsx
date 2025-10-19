import React, { useMemo } from 'react'

// Simple client-side risk heatmap derived from recent P&L variance
export default function RiskHeatmap({ data }) {
  // Make a grid e.g., 10x10 using last 100 points
  const grid = useMemo(() => {
    const last = (data || []).slice(-100).map(d => Number(d.pnl) || 0)
    const mean = last.length ? last.reduce((a,b) => a+b, 0) / last.length : 0
    const variance = last.length ? last.reduce((a,b) => a + Math.pow(b-mean,2), 0)/last.length : 0
    const std = Math.sqrt(variance)

    // Normalize each cell value from -3..+3 std devs to color scale
    const norm = last.map(v => std ? (v - mean) / std : 0)
    const cells = new Array(100).fill(0).map((_, i) => norm[i] ?? 0)
    return new Array(10).fill(0).map((_, r) => cells.slice(r*10, r*10+10))
  }, [data])

  const colorFor = (z) => {
    // Map -3..0..+3 to colors (red -> yellow -> green)
    const t = Math.max(-3, Math.min(3, z))
    if (t < -1) return '#c62828' // deep red
    if (t < 0) return '#ef6c00'  // orange
    if (t < 1) return '#fdd835'  // yellow
    if (t < 2) return '#7cb342'  // light green
    return '#2e7d32'             // green
  }

  return (
    <div className="heatmap">
      {grid.map((row, ri) => (
        <div key={ri} className="heatmap-row">
          {row.map((z, ci) => (
            <div key={ci} className="heatmap-cell" style={{ backgroundColor: colorFor(z) }} />
          ))}
        </div>
      ))}
    </div>
  )
}

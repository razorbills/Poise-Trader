import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple


@dataclass
class RiskEngineConfig:
    daily_loss_stop_pct: float = 0.10
    weekly_loss_stop_pct: float = 0.20
    max_drawdown_pct: float = 0.25
    max_total_notional_multiple: float = 3.0
    max_symbol_notional_multiple: float = 1.2
    loss_streak_pause_trades: int = 4
    loss_streak_reduce_trades: int = 2
    loss_streak_pause_seconds: float = 900.0


class RiskEngineV2:
    def __init__(self, initial_equity: float, config: Optional[RiskEngineConfig] = None):
        self.enabled = True
        try:
            env = str(os.getenv('RISK_ENGINE_ENABLED', '1') or '1').strip().lower()
            self.enabled = env in ['1', 'true', 'yes', 'on']
        except Exception:
            self.enabled = True

        self.config = config or self._config_from_env(initial_equity)

        self.initial_equity = float(initial_equity or 0.0)
        self.daily_start_equity = float(initial_equity or 0.0)
        self.weekly_start_equity = float(initial_equity or 0.0)
        self.peak_equity = float(initial_equity or 0.0)
        self.last_equity = float(initial_equity or 0.0)

        self.consecutive_losses = 0
        self.pause_until_ts = 0.0
        self.pause_reason = None

        self.last_daily_reset_key = None
        self.last_weekly_reset_key = None

        self.last_update_ts = 0.0
        self.last_state = {
            'allowed': True,
            'risk_multiplier': 1.0,
            'reason': None,
            'daily_pnl_pct': 0.0,
            'weekly_pnl_pct': 0.0,
            'drawdown_pct': 0.0,
        }

    def _config_from_env(self, initial_equity: float) -> RiskEngineConfig:
        cfg = RiskEngineConfig()

        try:
            profile = str(os.getenv('RISK_ENGINE_PROFILE', '') or '').strip().lower()
        except Exception:
            profile = ''

        if initial_equity and initial_equity < 50:
            cfg.daily_loss_stop_pct = 0.20
            cfg.weekly_loss_stop_pct = 0.35
            cfg.max_drawdown_pct = 0.35
            cfg.max_total_notional_multiple = 2.5
            cfg.max_symbol_notional_multiple = 1.1
        elif initial_equity and initial_equity < 500:
            cfg.daily_loss_stop_pct = 0.10
            cfg.weekly_loss_stop_pct = 0.25
            cfg.max_drawdown_pct = 0.25
            cfg.max_total_notional_multiple = 3.0
            cfg.max_symbol_notional_multiple = 1.2
        else:
            cfg.daily_loss_stop_pct = 0.05
            cfg.weekly_loss_stop_pct = 0.15
            cfg.max_drawdown_pct = 0.20
            cfg.max_total_notional_multiple = 3.5
            cfg.max_symbol_notional_multiple = 1.5

        if profile in ['world_class', 'wc']:
            cfg.max_total_notional_multiple = min(cfg.max_total_notional_multiple, 2.5)
            cfg.max_symbol_notional_multiple = min(cfg.max_symbol_notional_multiple, 1.1)
            cfg.loss_streak_reduce_trades = 2
            cfg.loss_streak_pause_trades = 3
            cfg.loss_streak_pause_seconds = 1200.0

        try:
            cfg.daily_loss_stop_pct = float(os.getenv('RISK_DAILY_LOSS_STOP_PCT', str(cfg.daily_loss_stop_pct)) or cfg.daily_loss_stop_pct)
        except Exception:
            pass
        try:
            cfg.weekly_loss_stop_pct = float(os.getenv('RISK_WEEKLY_LOSS_STOP_PCT', str(cfg.weekly_loss_stop_pct)) or cfg.weekly_loss_stop_pct)
        except Exception:
            pass
        try:
            cfg.max_drawdown_pct = float(os.getenv('RISK_MAX_DRAWDOWN_PCT', str(cfg.max_drawdown_pct)) or cfg.max_drawdown_pct)
        except Exception:
            pass

        return cfg

    def _utc_day_key(self, ts: float) -> str:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return dt.strftime('%Y-%m-%d')

    def _utc_week_key(self, ts: float) -> str:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        year, week, _ = dt.isocalendar()
        return f'{year}-W{week:02d}'

    def update_equity(self, equity: float, now_ts: Optional[float] = None) -> Dict[str, Any]:
        if not self.enabled:
            return dict(self.last_state)

        now_ts = float(now_ts or time.time())
        equity = float(equity or 0.0)

        day_key = self._utc_day_key(now_ts)
        if self.last_daily_reset_key is None:
            self.last_daily_reset_key = day_key
        if day_key != self.last_daily_reset_key:
            self.daily_start_equity = equity
            self.last_daily_reset_key = day_key
            self.consecutive_losses = 0

        week_key = self._utc_week_key(now_ts)
        if self.last_weekly_reset_key is None:
            self.last_weekly_reset_key = week_key
        if week_key != self.last_weekly_reset_key:
            self.weekly_start_equity = equity
            self.last_weekly_reset_key = week_key

        self.last_equity = equity
        if equity > self.peak_equity:
            self.peak_equity = equity

        daily_pnl_pct = 0.0
        weekly_pnl_pct = 0.0
        drawdown_pct = 0.0

        try:
            if self.daily_start_equity > 0:
                daily_pnl_pct = (equity - self.daily_start_equity) / self.daily_start_equity
        except Exception:
            daily_pnl_pct = 0.0
        try:
            if self.weekly_start_equity > 0:
                weekly_pnl_pct = (equity - self.weekly_start_equity) / self.weekly_start_equity
        except Exception:
            weekly_pnl_pct = 0.0
        try:
            if self.peak_equity > 0:
                drawdown_pct = max(0.0, (self.peak_equity - equity) / self.peak_equity)
        except Exception:
            drawdown_pct = 0.0

        allowed = True
        reason = None
        risk_multiplier = 1.0

        if self.pause_until_ts and now_ts < self.pause_until_ts:
            allowed = False
            reason = self.pause_reason or 'paused'

        if allowed:
            if daily_pnl_pct <= -abs(self.config.daily_loss_stop_pct):
                allowed = False
                reason = 'daily_loss_stop'
            elif weekly_pnl_pct <= -abs(self.config.weekly_loss_stop_pct):
                allowed = False
                reason = 'weekly_loss_stop'
            elif drawdown_pct >= abs(self.config.max_drawdown_pct):
                allowed = False
                reason = 'max_drawdown_stop'

        if allowed:
            if daily_pnl_pct < 0:
                risk_multiplier *= max(0.2, 1.0 + daily_pnl_pct / max(0.0001, abs(self.config.daily_loss_stop_pct)))
            if drawdown_pct > 0:
                risk_multiplier *= max(0.3, 1.0 - (drawdown_pct / max(0.0001, abs(self.config.max_drawdown_pct))))

            if self.consecutive_losses >= int(self.config.loss_streak_pause_trades):
                allowed = False
                reason = 'loss_streak_pause'
            elif self.consecutive_losses >= int(self.config.loss_streak_reduce_trades):
                risk_multiplier *= 0.5

        risk_multiplier = max(0.0, min(1.0, float(risk_multiplier)))

        self.last_state = {
            'allowed': bool(allowed),
            'risk_multiplier': float(risk_multiplier),
            'reason': reason,
            'daily_pnl_pct': float(daily_pnl_pct),
            'weekly_pnl_pct': float(weekly_pnl_pct),
            'drawdown_pct': float(drawdown_pct),
            'pause_until_ts': float(self.pause_until_ts or 0.0),
        }
        self.last_update_ts = now_ts
        return dict(self.last_state)

    def on_trade_closed(self, pnl: float, now_ts: Optional[float] = None):
        if not self.enabled:
            return

        pnl = float(pnl or 0.0)
        if pnl < 0:
            self.consecutive_losses = int(self.consecutive_losses or 0) + 1
        else:
            self.consecutive_losses = 0

        if self.consecutive_losses >= int(self.config.loss_streak_pause_trades):
            now_ts = float(now_ts or time.time())
            self.pause_until_ts = now_ts + float(self.config.loss_streak_pause_seconds)
            self.pause_reason = 'loss_streak_pause'

    def check_entry(
        self,
        symbol: str,
        equity: float,
        positions: Dict[str, Any],
        proposed_margin_usd: float,
        leverage: float,
    ) -> Tuple[bool, float, Optional[str]]:
        if not self.enabled:
            return True, 1.0, None

        equity = float(equity or 0.0)
        leverage = float(leverage or 1.0)
        if leverage <= 0:
            leverage = 1.0

        state = self.update_equity(equity)
        if not state.get('allowed', True):
            return False, 0.0, state.get('reason')

        risk_multiplier = float(state.get('risk_multiplier', 1.0) or 1.0)

        total_notional = 0.0
        symbol_notional = 0.0
        try:
            for sym, pos in (positions or {}).items():
                if not isinstance(pos, dict):
                    continue
                cv = float(pos.get('current_value', 0) or 0)
                if cv <= 0:
                    continue
                total_notional += cv
                if str(sym) == str(symbol):
                    symbol_notional += cv
        except Exception:
            total_notional = 0.0
            symbol_notional = 0.0

        proposed_margin_usd = float(proposed_margin_usd or 0.0)
        proposed_notional = proposed_margin_usd * leverage

        if equity > 0:
            max_total = float(self.config.max_total_notional_multiple) * equity
            max_sym = float(self.config.max_symbol_notional_multiple) * equity

            if (total_notional + proposed_notional) > max_total:
                return False, 0.0, 'total_exposure_cap'
            if (symbol_notional + proposed_notional) > max_sym:
                return False, 0.0, 'symbol_exposure_cap'

        return True, risk_multiplier, None

    def get_state(self) -> Dict[str, Any]:
        return dict(self.last_state)

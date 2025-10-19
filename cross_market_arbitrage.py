"""
💰 CROSS-MARKET ARBITRAGE ENGINE 💰
Crypto ↔ Forex ↔ Commodities arbitrage opportunities
Instant hedging when BTC tanks and gold pumps
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class MarketType(Enum):
    CRYPTO = "crypto"
    FOREX = "forex"
    COMMODITIES = "commodities"
    STOCKS = "stocks"

@dataclass
class ArbitrageOpportunity:
    """Cross-market arbitrage opportunity"""
    asset_pair: str
    market_a: MarketType
    market_b: MarketType
    price_a: float
    price_b: float
    spread_pct: float
    volume_required: float
    estimated_profit: float
    risk_score: float
    expiry_time: datetime
    correlation_strength: float

@dataclass
class HedgeSignal:
    """Hedging signal when one market moves"""
    trigger_asset: str
    trigger_market: MarketType
    hedge_asset: str
    hedge_market: MarketType
    hedge_direction: str  # buy/sell
    hedge_ratio: float
    confidence: float
    urgency: str  # low/medium/high/critical

class CrossMarketArbitrage:
    """Cross-market arbitrage engine for multiple asset classes"""
    
    def __init__(self):
        self.market_data = {
            MarketType.CRYPTO: {},
            MarketType.FOREX: {},
            MarketType.COMMODITIES: {},
            MarketType.STOCKS: {}
        }
        self.correlation_matrix = {}
        self.arbitrage_opportunities = []
        self.hedge_signals = []
        self.historical_correlations = {}
        
        # Key correlation pairs for instant hedging
        self.hedge_pairs = {
            'BTC_GOLD': {'correlation': -0.3, 'hedge_ratio': 0.15},
            'BTC_USD': {'correlation': -0.4, 'hedge_ratio': 0.20},
            'ETH_NASDAQ': {'correlation': 0.6, 'hedge_ratio': 0.25},
            'CRYPTO_VIX': {'correlation': -0.5, 'hedge_ratio': 0.30},
            'OIL_USD': {'correlation': -0.7, 'hedge_ratio': 0.40}
        }
        
    async def update_all_markets(self):
        """Update data from all markets simultaneously"""
        await asyncio.gather(
            self._update_crypto_data(),
            self._update_forex_data(),
            self._update_commodities_data(),
            self._update_stocks_data()
        )
        
        # Update correlations after data refresh
        await self._update_correlations()
    
    async def _update_crypto_data(self):
        """Update crypto market data"""
        # Simulated crypto data with realistic volatility
        base_prices = {'BTC/USD': 65000, 'ETH/USD': 3500, 'BNB/USD': 600, 'SOL/USD': 150}
        
        self.market_data[MarketType.CRYPTO] = {}
        for symbol, base_price in base_prices.items():
            volatility = 0.02 if 'BTC' in symbol else 0.03
            price_change = np.random.normal(0, base_price * volatility)
            
            self.market_data[MarketType.CRYPTO][symbol] = {
                'price': base_price + price_change,
                'volume': np.random.uniform(500000, 2000000),
                'change_24h': price_change / base_price,
                'timestamp': datetime.now()
            }
    
    async def _update_forex_data(self):
        """Update forex market data"""
        base_rates = {'EUR/USD': 1.0850, 'GBP/USD': 1.2650, 'USD/JPY': 149.50, 'USD/CHF': 0.8950}
        
        self.market_data[MarketType.FOREX] = {}
        for pair, base_rate in base_rates.items():
            volatility = 0.005  # Forex is less volatile
            rate_change = np.random.normal(0, base_rate * volatility)
            
            self.market_data[MarketType.FOREX][pair] = {
                'price': base_rate + rate_change,
                'volume': np.random.uniform(3000000, 8000000),
                'change_24h': rate_change / base_rate,
                'timestamp': datetime.now()
            }
    
    async def _update_commodities_data(self):
        """Update commodities market data"""
        base_prices = {'GOLD': 2050, 'SILVER': 24.50, 'OIL': 85.00, 'COPPER': 3.80}
        
        self.market_data[MarketType.COMMODITIES] = {}
        for commodity, base_price in base_prices.items():
            volatility = 0.015 if commodity == 'GOLD' else 0.025
            price_change = np.random.normal(0, base_price * volatility)
            
            self.market_data[MarketType.COMMODITIES][commodity] = {
                'price': base_price + price_change,
                'volume': np.random.uniform(50000, 300000),
                'change_24h': price_change / base_price,
                'timestamp': datetime.now()
            }
    
    async def _update_stocks_data(self):
        """Update stock market data"""
        base_prices = {'SPY': 450, 'QQQ': 380, 'GLD': 185, 'VIX': 18.5}
        
        self.market_data[MarketType.STOCKS] = {}
        for symbol, base_price in base_prices.items():
            volatility = 0.01 if symbol != 'VIX' else 0.05
            price_change = np.random.normal(0, base_price * volatility)
            
            self.market_data[MarketType.STOCKS][symbol] = {
                'price': base_price + price_change,
                'volume': np.random.uniform(800000, 2000000),
                'change_24h': price_change / base_price,
                'timestamp': datetime.now()
            }
    
    async def _update_correlations(self):
        """Update correlation matrix between assets"""
        # Calculate real-time correlations (simplified)
        btc_change = self.market_data[MarketType.CRYPTO].get('BTC/USD', {}).get('change_24h', 0)
        gold_change = self.market_data[MarketType.COMMODITIES].get('GOLD', {}).get('change_24h', 0)
        usd_strength = -self.market_data[MarketType.FOREX].get('EUR/USD', {}).get('change_24h', 0)
        
        # Update correlation tracking
        self.correlation_matrix = {
            'BTC_GOLD': -0.3 + np.random.normal(0, 0.1),
            'BTC_USD': -0.4 + np.random.normal(0, 0.1),
            'GOLD_USD': 0.7 + np.random.normal(0, 0.05),
            'CRYPTO_STOCKS': 0.5 + np.random.normal(0, 0.1)
        }
    
    async def find_arbitrage_opportunities(self) -> List[ArbitrageOpportunity]:
        """Find cross-market arbitrage opportunities"""
        opportunities = []
        
        # BTC vs Gold arbitrage
        btc_data = self.market_data[MarketType.CRYPTO].get('BTC/USD', {})
        gold_data = self.market_data[MarketType.COMMODITIES].get('GOLD', {})
        
        if btc_data and gold_data:
            btc_price = btc_data['price']
            gold_price = gold_data['price']
            
            # Historical BTC/Gold ratio analysis
            historical_ratio = 31.7  # BTC typically trades at ~32x gold price
            current_ratio = btc_price / gold_price
            deviation = abs(current_ratio - historical_ratio) / historical_ratio
            
            if deviation > 0.03:  # 3% deviation threshold
                spread_pct = (current_ratio - historical_ratio) / historical_ratio * 100
                
                opportunity = ArbitrageOpportunity(
                    asset_pair="BTC/GOLD",
                    market_a=MarketType.CRYPTO,
                    market_b=MarketType.COMMODITIES,
                    price_a=btc_price,
                    price_b=gold_price,
                    spread_pct=abs(spread_pct),
                    volume_required=15000,  # $15k minimum
                    estimated_profit=abs(spread_pct) * 150,
                    risk_score=0.25,
                    expiry_time=datetime.now() + timedelta(hours=2),
                    correlation_strength=abs(self.correlation_matrix.get('BTC_GOLD', -0.3))
                )
                opportunities.append(opportunity)
        
        # EUR/USD vs BTC correlation arbitrage
        eur_data = self.market_data[MarketType.FOREX].get('EUR/USD', {})
        if btc_data and eur_data:
            eur_usd = eur_data['price']
            btc_change = btc_data['change_24h']
            eur_change = eur_data['change_24h']
            
            # When EUR weakens significantly, crypto often strengthens
            if eur_change < -0.005 and btc_change > 0.01:  # Inverse correlation opportunity
                opportunity = ArbitrageOpportunity(
                    asset_pair="BTC/EUR",
                    market_a=MarketType.CRYPTO,
                    market_b=MarketType.FOREX,
                    price_a=btc_data['price'],
                    price_b=eur_usd,
                    spread_pct=3.2,
                    volume_required=8000,
                    estimated_profit=256,
                    risk_score=0.35,
                    expiry_time=datetime.now() + timedelta(minutes=45),
                    correlation_strength=0.4
                )
                opportunities.append(opportunity)
        
        # Oil vs USD strength arbitrage
        oil_data = self.market_data[MarketType.COMMODITIES].get('OIL', {})
        if oil_data and eur_data:
            oil_change = oil_data['change_24h']
            usd_strength = -eur_data['change_24h']  # Inverse of EUR/USD
            
            # Oil and USD often move inversely
            if oil_change < -0.02 and usd_strength > 0.003:
                opportunity = ArbitrageOpportunity(
                    asset_pair="OIL/USD",
                    market_a=MarketType.COMMODITIES,
                    market_b=MarketType.FOREX,
                    price_a=oil_data['price'],
                    price_b=eur_data['price'],
                    spread_pct=2.8,
                    volume_required=12000,
                    estimated_profit=336,
                    risk_score=0.30,
                    expiry_time=datetime.now() + timedelta(hours=1),
                    correlation_strength=0.7
                )
                opportunities.append(opportunity)
        
        self.arbitrage_opportunities = opportunities
        return opportunities
    
    async def detect_hedge_signals(self) -> List[HedgeSignal]:
        """Detect when instant hedging is needed"""
        signals = []
        
        # BTC tanks → Gold hedge
        btc_data = self.market_data[MarketType.CRYPTO].get('BTC/USD', {})
        gold_data = self.market_data[MarketType.COMMODITIES].get('GOLD', {})
        
        if btc_data and gold_data:
            btc_change = btc_data['change_24h']
            
            # If BTC drops more than 3%, hedge with gold
            if btc_change < -0.03:
                urgency = 'critical' if btc_change < -0.05 else 'high'
                
                signal = HedgeSignal(
                    trigger_asset='BTC/USD',
                    trigger_market=MarketType.CRYPTO,
                    hedge_asset='GOLD',
                    hedge_market=MarketType.COMMODITIES,
                    hedge_direction='buy',
                    hedge_ratio=min(0.4, abs(btc_change) * 10),  # Scale with BTC drop
                    confidence=0.8,
                    urgency=urgency
                )
                signals.append(signal)
        
        # Crypto correlation with NASDAQ
        eth_data = self.market_data[MarketType.CRYPTO].get('ETH/USD', {})
        qqq_data = self.market_data[MarketType.STOCKS].get('QQQ', {})
        
        if eth_data and qqq_data:
            eth_change = eth_data['change_24h']
            qqq_change = qqq_data['change_24h']
            
            # If correlation breaks down (ETH up, NASDAQ down), hedge
            if eth_change > 0.02 and qqq_change < -0.015:
                signal = HedgeSignal(
                    trigger_asset='ETH/USD',
                    trigger_market=MarketType.CRYPTO,
                    hedge_asset='QQQ',
                    hedge_market=MarketType.STOCKS,
                    hedge_direction='sell',
                    hedge_ratio=0.25,
                    confidence=0.7,
                    urgency='medium'
                )
                signals.append(signal)
        
        # VIX spike → Crypto hedge
        vix_data = self.market_data[MarketType.STOCKS].get('VIX', {})
        if vix_data and btc_data:
            vix_price = vix_data['price']
            
            # If VIX spikes above 25, hedge crypto positions
            if vix_price > 25:
                urgency = 'critical' if vix_price > 30 else 'high'
                
                signal = HedgeSignal(
                    trigger_asset='VIX',
                    trigger_market=MarketType.STOCKS,
                    hedge_asset='BTC/USD',
                    hedge_market=MarketType.CRYPTO,
                    hedge_direction='sell',
                    hedge_ratio=min(0.5, (vix_price - 20) / 20),
                    confidence=0.85,
                    urgency=urgency
                )
                signals.append(signal)
        
        self.hedge_signals = signals
        return signals
    
    async def execute_arbitrage(self, opportunity: ArbitrageOpportunity) -> Dict[str, Any]:
        """Execute cross-market arbitrage trade"""
        print(f"⚡ EXECUTING ARBITRAGE: {opportunity.asset_pair}")
        print(f"   💰 Spread: {opportunity.spread_pct:.2f}%")
        print(f"   🎯 Est. Profit: ${opportunity.estimated_profit:.2f}")
        
        # Simulated execution
        execution_result = {
            'success': True,
            'executed_volume': opportunity.volume_required,
            'actual_profit': opportunity.estimated_profit * np.random.uniform(0.8, 1.2),
            'execution_time': datetime.now(),
            'slippage': np.random.uniform(0.001, 0.003),
            'fees': opportunity.volume_required * 0.001  # 0.1% fees
        }
        
        net_profit = execution_result['actual_profit'] - execution_result['fees']
        print(f"   ✅ Net Profit: ${net_profit:.2f}")
        
        return execution_result
    
    async def execute_hedge(self, signal: HedgeSignal) -> Dict[str, Any]:
        """Execute instant hedge trade"""
        print(f"🛡️ EXECUTING HEDGE: {signal.trigger_asset} → {signal.hedge_asset}")
        print(f"   📊 Hedge Ratio: {signal.hedge_ratio:.1%}")
        print(f"   🚨 Urgency: {signal.urgency.upper()}")
        
        # Calculate hedge size based on current portfolio
        portfolio_value = 10000  # Simulated portfolio value
        hedge_size = portfolio_value * signal.hedge_ratio
        
        execution_result = {
            'success': True,
            'hedge_direction': signal.hedge_direction,
            'hedge_size': hedge_size,
            'execution_time': datetime.now(),
            'confidence': signal.confidence,
            'urgency': signal.urgency
        }
        
        print(f"   💼 Hedge Size: ${hedge_size:.2f}")
        print(f"   ✅ Hedge executed successfully")
        
        return execution_result
    
    def get_correlation_strength(self, asset1: str, asset2: str) -> float:
        """Get correlation strength between two assets"""
        pair_key = f"{asset1}_{asset2}"
        reverse_key = f"{asset2}_{asset1}"
        
        return self.correlation_matrix.get(pair_key, 
               self.correlation_matrix.get(reverse_key, 0.0))
    
    async def monitor_cross_market_flows(self) -> Dict[str, Any]:
        """Monitor capital flows between markets"""
        flows = {
            'crypto_to_gold': 0,
            'forex_to_crypto': 0,
            'stocks_to_commodities': 0,
            'total_arbitrage_volume': sum(op.volume_required for op in self.arbitrage_opportunities),
            'active_hedges': len(self.hedge_signals),
            'market_stress_level': self._calculate_market_stress()
        }
        
        # Detect unusual flows
        if flows['total_arbitrage_volume'] > 50000:
            flows['flow_alert'] = 'High arbitrage activity detected'
        
        if flows['active_hedges'] > 3:
            flows['hedge_alert'] = 'Multiple hedge signals active'
        
        return flows
    
    def _calculate_market_stress(self) -> float:
        """Calculate overall market stress level"""
        stress_factors = []
        
        # VIX level
        vix_data = self.market_data[MarketType.STOCKS].get('VIX', {})
        if vix_data:
            vix_stress = min(1.0, (vix_data['price'] - 15) / 20)  # Normalize VIX
            stress_factors.append(vix_stress)
        
        # Crypto volatility
        btc_data = self.market_data[MarketType.CRYPTO].get('BTC/USD', {})
        if btc_data:
            btc_stress = min(1.0, abs(btc_data['change_24h']) / 0.1)  # 10% change = max stress
            stress_factors.append(btc_stress)
        
        # Correlation breakdown
        correlation_stress = 1 - np.mean([abs(corr) for corr in self.correlation_matrix.values()])
        stress_factors.append(correlation_stress)
        
        return np.mean(stress_factors) if stress_factors else 0.0
    
    async def detect_opportunities(self) -> List[ArbitrageOpportunity]:
        """Detect arbitrage opportunities (alias for find_arbitrage_opportunities)"""
        return await self.find_arbitrage_opportunities()

#!/usr/bin/env python3
"""
⚖️ REGULATORY COMPLIANCE & AUDIT TRAIL SYSTEM
Trade Reporting, Best Execution, Record Keeping & Tax Optimization
"""

import asyncio
import json
import hashlib
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from decimal import Decimal, ROUND_HALF_UP
import logging
from enum import Enum
import csv
import os

@dataclass
class TradeRecord:
    """Comprehensive trade record for regulatory compliance"""
    trade_id: str
    timestamp: datetime
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    price: float
    total_value: float
    fees: float
    exchange: str
    order_type: str
    execution_venue: str
    client_order_id: str
    market_order_id: str
    liquidity_flag: str  # 'MAKER' or 'TAKER'
    slippage_bps: float
    latency_ms: float
    regulatory_flags: List[str]
    best_execution_score: float
    tax_lot_method: str
    trade_hash: str

@dataclass
class PositionRecord:
    """Position tracking for compliance"""
    position_id: str
    symbol: str
    entry_timestamp: datetime
    exit_timestamp: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    realized_pnl: Optional[float]
    unrealized_pnl: float
    holding_period_days: Optional[int]
    tax_treatment: str  # 'SHORT_TERM', 'LONG_TERM'
    wash_sale_flag: bool

class TaxLotMethod(Enum):
    """Tax lot accounting methods"""
    FIFO = "First In, First Out"
    LIFO = "Last In, First Out"
    HIFO = "Highest Cost First Out"
    SPECIFIC_ID = "Specific Identification"
    AVERAGE_COST = "Average Cost"

class ComplianceDatabase:
    """SQLite database for compliance records"""
    
    def __init__(self, db_path: str = "compliance.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize compliance database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Trade records table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trade_records (
                trade_id TEXT PRIMARY KEY,
                timestamp TEXT,
                symbol TEXT,
                side TEXT,
                quantity REAL,
                price REAL,
                total_value REAL,
                fees REAL,
                exchange TEXT,
                order_type TEXT,
                execution_venue TEXT,
                client_order_id TEXT,
                market_order_id TEXT,
                liquidity_flag TEXT,
                slippage_bps REAL,
                latency_ms REAL,
                regulatory_flags TEXT,
                best_execution_score REAL,
                tax_lot_method TEXT,
                trade_hash TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Position records table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS position_records (
                position_id TEXT PRIMARY KEY,
                symbol TEXT,
                entry_timestamp TEXT,
                exit_timestamp TEXT,
                entry_price REAL,
                exit_price REAL,
                quantity REAL,
                realized_pnl REAL,
                unrealized_pnl REAL,
                holding_period_days INTEGER,
                tax_treatment TEXT,
                wash_sale_flag BOOLEAN,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Audit log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_log (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                event_type TEXT,
                user_id TEXT,
                action TEXT,
                details TEXT,
                ip_address TEXT,
                session_id TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Best execution analysis table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS best_execution_analysis (
                analysis_id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT,
                symbol TEXT,
                timestamp TEXT,
                execution_venue TEXT,
                execution_price REAL,
                benchmark_price REAL,
                price_improvement_bps REAL,
                effective_spread_bps REAL,
                market_impact_bps REAL,
                timing_cost_bps REAL,
                total_cost_bps REAL,
                venue_ranking INTEGER,
                FOREIGN KEY (trade_id) REFERENCES trade_records (trade_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_trade_record(self, trade: TradeRecord):
        """Store trade record in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO trade_records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade.trade_id, trade.timestamp.isoformat(), trade.symbol, trade.side,
            trade.quantity, trade.price, trade.total_value, trade.fees,
            trade.exchange, trade.order_type, trade.execution_venue,
            trade.client_order_id, trade.market_order_id, trade.liquidity_flag,
            trade.slippage_bps, trade.latency_ms, json.dumps(trade.regulatory_flags),
            trade.best_execution_score, trade.tax_lot_method, trade.trade_hash
        ))
        
        conn.commit()
        conn.close()
    
    def get_trades_by_period(self, start_date: datetime, end_date: datetime) -> List[TradeRecord]:
        """Retrieve trades for a specific period"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM trade_records 
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        ''', (start_date.isoformat(), end_date.isoformat()))
        
        trades = []
        for row in cursor.fetchall():
            trade = TradeRecord(
                trade_id=row[0], timestamp=datetime.fromisoformat(row[1]),
                symbol=row[2], side=row[3], quantity=row[4], price=row[5],
                total_value=row[6], fees=row[7], exchange=row[8],
                order_type=row[9], execution_venue=row[10], client_order_id=row[11],
                market_order_id=row[12], liquidity_flag=row[13], slippage_bps=row[14],
                latency_ms=row[15], regulatory_flags=json.loads(row[16]),
                best_execution_score=row[17], tax_lot_method=row[18], trade_hash=row[19]
            )
            trades.append(trade)
        
        conn.close()
        return trades

class BestExecutionAnalyzer:
    """Best execution analysis and reporting"""
    
    def __init__(self, db: ComplianceDatabase):
        self.db = db
        
    def analyze_execution_quality(self, trade: TradeRecord, market_data: Dict) -> Dict:
        """Analyze execution quality against best execution standards"""
        
        # Get benchmark prices from multiple venues
        benchmark_prices = market_data.get('venue_prices', {})
        
        if not benchmark_prices:
            return {'error': 'No benchmark data available'}
        
        execution_price = trade.price
        
        # Calculate best available price
        if trade.side == 'BUY':
            best_price = min(benchmark_prices.values())
            price_improvement = (best_price - execution_price) / execution_price * 10000
        else:  # SELL
            best_price = max(benchmark_prices.values())
            price_improvement = (execution_price - best_price) / best_price * 10000
        
        # Calculate effective spread
        bid_ask_spread = market_data.get('spread_bps', 0)
        effective_spread = abs(execution_price - market_data.get('mid_price', execution_price)) / market_data.get('mid_price', execution_price) * 10000
        
        # Market impact estimation
        volume_ratio = trade.quantity / market_data.get('avg_volume', trade.quantity)
        market_impact = min(50, volume_ratio * 25)  # Cap at 50 bps
        
        # Timing cost (latency impact)
        timing_cost = trade.latency_ms / 100 * 2  # 2 bps per 100ms
        
        # Total cost calculation
        total_cost = abs(price_improvement) + effective_spread + market_impact + timing_cost
        
        analysis = {
            'trade_id': trade.trade_id,
            'execution_venue': trade.execution_venue,
            'execution_price': execution_price,
            'benchmark_price': best_price,
            'price_improvement_bps': price_improvement,
            'effective_spread_bps': effective_spread,
            'market_impact_bps': market_impact,
            'timing_cost_bps': timing_cost,
            'total_cost_bps': total_cost,
            'execution_quality_score': max(0, 100 - total_cost),
            'venue_ranking': self._rank_execution_venue(trade.execution_venue, benchmark_prices)
        }
        
        # Store analysis
        self._store_execution_analysis(analysis)
        
        return analysis
    
    def _rank_execution_venue(self, venue: str, venue_prices: Dict) -> int:
        """Rank execution venue by price competitiveness"""
        prices = list(venue_prices.values())
        if venue not in venue_prices:
            return len(prices) + 1
        
        venue_price = venue_prices[venue]
        better_prices = sum(1 for p in prices if p < venue_price)
        return better_prices + 1
    
    def _store_execution_analysis(self, analysis: Dict):
        """Store execution analysis in database"""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO best_execution_analysis 
            (trade_id, symbol, timestamp, execution_venue, execution_price, benchmark_price,
             price_improvement_bps, effective_spread_bps, market_impact_bps, timing_cost_bps,
             total_cost_bps, venue_ranking)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            analysis['trade_id'], '', datetime.now().isoformat(),
            analysis['execution_venue'], analysis['execution_price'], analysis['benchmark_price'],
            analysis['price_improvement_bps'], analysis['effective_spread_bps'],
            analysis['market_impact_bps'], analysis['timing_cost_bps'],
            analysis['total_cost_bps'], analysis['venue_ranking']
        ))
        
        conn.commit()
        conn.close()

class TaxOptimizer:
    """Tax optimization and lot tracking system"""
    
    def __init__(self, db: ComplianceDatabase):
        self.db = db
        self.tax_lots = {}  # symbol -> list of tax lots
        self.wash_sale_tracker = {}
        
    def add_tax_lot(self, symbol: str, quantity: float, price: float, timestamp: datetime, trade_id: str):
        """Add a new tax lot for tracking"""
        if symbol not in self.tax_lots:
            self.tax_lots[symbol] = []
        
        tax_lot = {
            'quantity': quantity,
            'price': price,
            'timestamp': timestamp,
            'trade_id': trade_id,
            'remaining_quantity': quantity
        }
        
        self.tax_lots[symbol].append(tax_lot)
    
    def optimize_tax_lot_selection(self, symbol: str, quantity: float, current_price: float, method: TaxLotMethod = TaxLotMethod.HIFO) -> List[Dict]:
        """Select optimal tax lots for sale to minimize tax liability"""
        
        if symbol not in self.tax_lots:
            return []
        
        available_lots = [lot for lot in self.tax_lots[symbol] if lot['remaining_quantity'] > 0]
        
        if method == TaxLotMethod.FIFO:
            available_lots.sort(key=lambda x: x['timestamp'])
        elif method == TaxLotMethod.LIFO:
            available_lots.sort(key=lambda x: x['timestamp'], reverse=True)
        elif method == TaxLotMethod.HIFO:
            available_lots.sort(key=lambda x: x['price'], reverse=True)
        
        selected_lots = []
        remaining_quantity = quantity
        
        for lot in available_lots:
            if remaining_quantity <= 0:
                break
            
            lot_quantity = min(lot['remaining_quantity'], remaining_quantity)
            
            # Calculate gain/loss
            gain_loss = (current_price - lot['price']) * lot_quantity
            
            # Determine holding period
            holding_days = (datetime.now() - lot['timestamp']).days
            tax_treatment = 'LONG_TERM' if holding_days > 365 else 'SHORT_TERM'
            
            selected_lot = {
                'original_lot': lot,
                'quantity_used': lot_quantity,
                'gain_loss': gain_loss,
                'tax_treatment': tax_treatment,
                'holding_days': holding_days
            }
            
            selected_lots.append(selected_lot)
            lot['remaining_quantity'] -= lot_quantity
            remaining_quantity -= lot_quantity
        
        return selected_lots
    
    def check_wash_sale(self, symbol: str, sale_date: datetime, loss_amount: float) -> bool:
        """Check for wash sale violations (30-day rule)"""
        
        # Check for purchases 30 days before or after sale
        wash_sale_window_start = sale_date - timedelta(days=30)
        wash_sale_window_end = sale_date + timedelta(days=30)
        
        for lot in self.tax_lots.get(symbol, []):
            if wash_sale_window_start <= lot['timestamp'] <= wash_sale_window_end:
                if lot['timestamp'] != sale_date:  # Not the same transaction
                    return True
        
        return False
    
    def generate_tax_report(self, tax_year: int) -> Dict:
        """Generate comprehensive tax report"""
        
        year_start = datetime(tax_year, 1, 1)
        year_end = datetime(tax_year, 12, 31, 23, 59, 59)
        
        trades = self.db.get_trades_by_period(year_start, year_end)
        
        short_term_gains = 0
        long_term_gains = 0
        total_volume = 0
        wash_sales = []
        
        for trade in trades:
            if trade.side == 'SELL':
                # Process sale using tax lot optimization
                selected_lots = self.optimize_tax_lot_selection(
                    trade.symbol, trade.quantity, trade.price
                )
                
                for lot_info in selected_lots:
                    if lot_info['tax_treatment'] == 'SHORT_TERM':
                        short_term_gains += lot_info['gain_loss']
                    else:
                        long_term_gains += lot_info['gain_loss']
                    
                    # Check for wash sales
                    if lot_info['gain_loss'] < 0:  # Loss
                        is_wash_sale = self.check_wash_sale(
                            trade.symbol, trade.timestamp, lot_info['gain_loss']
                        )
                        if is_wash_sale:
                            wash_sales.append({
                                'symbol': trade.symbol,
                                'trade_id': trade.trade_id,
                                'loss_amount': lot_info['gain_loss'],
                                'date': trade.timestamp
                            })
            
            total_volume += trade.total_value
        
        return {
            'tax_year': tax_year,
            'short_term_capital_gains': short_term_gains,
            'long_term_capital_gains': long_term_gains,
            'total_capital_gains': short_term_gains + long_term_gains,
            'total_trading_volume': total_volume,
            'wash_sales': wash_sales,
            'wash_sale_adjustments': sum(ws['loss_amount'] for ws in wash_sales),
            'net_capital_gains': short_term_gains + long_term_gains - abs(sum(ws['loss_amount'] for ws in wash_sales))
        }

class ComplianceReporter:
    """Generate regulatory compliance reports"""
    
    def __init__(self, db: ComplianceDatabase):
        self.db = db
    
    def generate_trade_report(self, start_date: datetime, end_date: datetime, format: str = 'csv') -> str:
        """Generate trade report for regulatory submission"""
        
        trades = self.db.get_trades_by_period(start_date, end_date)
        
        if format.lower() == 'csv':
            filename = f"trade_report_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
            
            with open(filename, 'w', newline='') as csvfile:
                fieldnames = [
                    'Trade ID', 'Timestamp', 'Symbol', 'Side', 'Quantity', 'Price',
                    'Total Value', 'Fees', 'Exchange', 'Order Type', 'Execution Venue',
                    'Client Order ID', 'Market Order ID', 'Liquidity Flag',
                    'Slippage (bps)', 'Latency (ms)', 'Best Execution Score'
                ]
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for trade in trades:
                    writer.writerow({
                        'Trade ID': trade.trade_id,
                        'Timestamp': trade.timestamp.isoformat(),
                        'Symbol': trade.symbol,
                        'Side': trade.side,
                        'Quantity': trade.quantity,
                        'Price': trade.price,
                        'Total Value': trade.total_value,
                        'Fees': trade.fees,
                        'Exchange': trade.exchange,
                        'Order Type': trade.order_type,
                        'Execution Venue': trade.execution_venue,
                        'Client Order ID': trade.client_order_id,
                        'Market Order ID': trade.market_order_id,
                        'Liquidity Flag': trade.liquidity_flag,
                        'Slippage (bps)': trade.slippage_bps,
                        'Latency (ms)': trade.latency_ms,
                        'Best Execution Score': trade.best_execution_score
                    })
            
            return filename
        
        return ""
    
    def generate_best_execution_report(self, period_days: int = 30) -> Dict:
        """Generate best execution analysis report"""
        
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)
        
        cursor.execute('''
            SELECT 
                execution_venue,
                COUNT(*) as trade_count,
                AVG(price_improvement_bps) as avg_price_improvement,
                AVG(effective_spread_bps) as avg_effective_spread,
                AVG(total_cost_bps) as avg_total_cost,
                AVG(venue_ranking) as avg_ranking
            FROM best_execution_analysis
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY execution_venue
            ORDER BY avg_total_cost ASC
        ''', (start_date.isoformat(), end_date.isoformat()))
        
        venue_analysis = []
        for row in cursor.fetchall():
            venue_analysis.append({
                'venue': row[0],
                'trade_count': row[1],
                'avg_price_improvement_bps': row[2],
                'avg_effective_spread_bps': row[3],
                'avg_total_cost_bps': row[4],
                'avg_ranking': row[5]
            })
        
        conn.close()
        
        return {
            'period_days': period_days,
            'analysis_date': datetime.now().isoformat(),
            'venue_performance': venue_analysis,
            'best_venue': venue_analysis[0]['venue'] if venue_analysis else None,
            'total_trades_analyzed': sum(v['trade_count'] for v in venue_analysis)
        }

class ComplianceManager:
    """Master compliance management system"""
    
    def __init__(self):
        self.db = ComplianceDatabase()
        self.best_execution = BestExecutionAnalyzer(self.db)
        self.tax_optimizer = TaxOptimizer(self.db)
        self.reporter = ComplianceReporter(self.db)
        
        # Compliance settings
        self.enable_best_execution_monitoring = True
        self.enable_tax_optimization = True
        self.enable_wash_sale_detection = True
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def process_trade(self, trade_data: Dict, market_data: Dict = None) -> TradeRecord:
        """Process trade through compliance pipeline"""
        
        # Create comprehensive trade record
        trade_record = TradeRecord(
            trade_id=trade_data.get('trade_id', self._generate_trade_id()),
            timestamp=datetime.now(),
            symbol=trade_data.get('symbol', ''),
            side=trade_data.get('side', ''),
            quantity=trade_data.get('quantity', 0),
            price=trade_data.get('price', 0),
            total_value=trade_data.get('quantity', 0) * trade_data.get('price', 0),
            fees=trade_data.get('fees', 0),
            exchange=trade_data.get('exchange', ''),
            order_type=trade_data.get('order_type', ''),
            execution_venue=trade_data.get('execution_venue', ''),
            client_order_id=trade_data.get('client_order_id', ''),
            market_order_id=trade_data.get('market_order_id', ''),
            liquidity_flag=trade_data.get('liquidity_flag', ''),
            slippage_bps=trade_data.get('slippage_bps', 0),
            latency_ms=trade_data.get('latency_ms', 0),
            regulatory_flags=[],
            best_execution_score=0,
            tax_lot_method=TaxLotMethod.HIFO.value,
            trade_hash=self._generate_trade_hash(trade_data)
        )
        
        # Best execution analysis
        if self.enable_best_execution_monitoring and market_data:
            execution_analysis = self.best_execution.analyze_execution_quality(trade_record, market_data)
            trade_record.best_execution_score = execution_analysis.get('execution_quality_score', 0)
        
        # Tax lot tracking
        if self.enable_tax_optimization:
            if trade_record.side == 'BUY':
                self.tax_optimizer.add_tax_lot(
                    trade_record.symbol, trade_record.quantity, 
                    trade_record.price, trade_record.timestamp, trade_record.trade_id
                )
            elif trade_record.side == 'SELL':
                # Optimize tax lot selection
                selected_lots = self.tax_optimizer.optimize_tax_lot_selection(
                    trade_record.symbol, trade_record.quantity, trade_record.price
                )
                
                # Check for wash sales
                for lot_info in selected_lots:
                    if lot_info['gain_loss'] < 0:
                        is_wash_sale = self.tax_optimizer.check_wash_sale(
                            trade_record.symbol, trade_record.timestamp, lot_info['gain_loss']
                        )
                        if is_wash_sale:
                            trade_record.regulatory_flags.append('WASH_SALE')
        
        # Store trade record
        self.db.store_trade_record(trade_record)
        
        self.logger.info(f"✅ Trade processed through compliance: {trade_record.trade_id}")
        
        return trade_record
    
    def _generate_trade_id(self) -> str:
        """Generate unique trade ID"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
        return f"TXN_{timestamp}"
    
    def _generate_trade_hash(self, trade_data: Dict) -> str:
        """Generate cryptographic hash for trade integrity"""
        trade_string = json.dumps(trade_data, sort_keys=True)
        return hashlib.sha256(trade_string.encode()).hexdigest()
    
    def get_compliance_summary(self) -> Dict:
        """Get comprehensive compliance status summary"""
        
        # Get recent compliance metrics
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        recent_trades = self.db.get_trades_by_period(start_date, end_date)
        
        # Best execution metrics
        best_execution_report = self.reporter.generate_best_execution_report(30)
        
        # Tax optimization metrics
        current_year = datetime.now().year
        tax_report = self.tax_optimizer.generate_tax_report(current_year)
        
        return {
            'compliance_status': 'ACTIVE',
            'monitoring_period_days': 30,
            'total_trades_monitored': len(recent_trades),
            'best_execution_enabled': self.enable_best_execution_monitoring,
            'tax_optimization_enabled': self.enable_tax_optimization,
            'wash_sale_detection_enabled': self.enable_wash_sale_detection,
            'best_execution_summary': best_execution_report,
            'tax_optimization_summary': tax_report,
            'regulatory_flags_count': sum(len(t.regulatory_flags) for t in recent_trades),
            'timestamp': datetime.now().isoformat()
        }

# Global compliance manager instance
compliance_manager = ComplianceManager()

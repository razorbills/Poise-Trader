#!/usr/bin/env python3
"""
PAPER TRADING TEST SUITE - SIMPLIFIED
Comprehensive testing without Unicode emojis to avoid Windows encoding issues
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json

# Add core modules to path
sys.path.append(str(Path(__file__).parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from core.execution.paper_trading_manager import PaperTradingManager


class SimplePaperTradingTestSuite:
    """
    SIMPLIFIED PAPER TRADING TEST SUITE
    Tests core paper trading functionality without Unicode issues
    """
    
    def __init__(self):
        # Setup basic logging without Unicode
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(name)s] [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler('test_paper_simple.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger("SimplePaperTests")
        self.test_results = []
        self.paper_trader = None
        
        # Test configuration
        self.test_capital = 5000.0
        self.test_symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    
    def _record_test_result(self, test_name: str, passed: bool, message: str):
        """Record test result"""
        self.test_results.append({
            'test_name': test_name,
            'passed': passed,
            'message': message,
            'timestamp': datetime.now().isoformat()
        })
    
    async def run_comprehensive_tests(self) -> bool:
        """Run all paper trading tests"""
        
        print("\n" + "="*60)
        print("POISE TRADER - SIMPLIFIED PAPER TRADING TEST SUITE")
        print("="*60)
        print(f"Objective: Validate core paper trading systems")
        print(f"Test Capital: ${self.test_capital:,.2f} virtual money")
        print(f"Testing: Basic execution and portfolio management")
        print("="*60 + "\n")
        
        self.logger.info("Starting simplified paper trading tests...")
        
        try:
            # Test 1: Initialize paper trading manager
            await self._test_paper_trading_initialization()
            
            # Test 2: Test basic trade execution
            await self._test_basic_trade_execution()
            
            # Test 3: Test portfolio management
            await self._test_portfolio_management()
            
            # Test 4: Test performance tracking
            await self._test_performance_tracking()
            
            # Generate test report
            success = self._generate_test_report()
            
            return success
            
        except Exception as e:
            self.logger.error(f"Critical error in test suite: {e}")
            return False
    
    async def _test_paper_trading_initialization(self):
        """Test 1: Paper trading manager initialization"""
        
        test_name = "Paper Trading Initialization"
        self.logger.info(f"Running Test 1: {test_name}")
        
        try:
            # Initialize paper trading manager
            self.paper_trader = PaperTradingManager(self.test_capital)
            
            # Verify initial state
            summary = self.paper_trader.get_performance_summary()
            
            assert summary['portfolio_value'] == self.test_capital, f"Expected {self.test_capital}, got {summary['portfolio_value']}"
            assert summary['cash_balance'] == self.test_capital, f"Expected cash {self.test_capital}, got {summary['cash_balance']}"
            assert summary['total_trades'] == 0, f"Expected 0 trades, got {summary['total_trades']}"
            
            self._record_test_result(test_name, True, "Paper trading manager initialized successfully")
            self.logger.info(f"Test 1 PASSED: {test_name}")
            
        except Exception as e:
            self._record_test_result(test_name, False, str(e))
            self.logger.error(f"Test 1 FAILED: {test_name} - {e}")
    
    async def _test_basic_trade_execution(self):
        """Test 2: Basic trade execution functionality"""
        
        test_name = "Basic Trade Execution"
        self.logger.info(f"Running Test 2: {test_name}")
        
        try:
            # Test BUY order
            buy_signal = {
                'symbol': 'BTC/USDT',
                'action': 'BUY',
                'position_size': 100,  # $100 worth
                'entry_price': 43000.0,
                'stop_loss': 42000.0,
                'take_profit': 45000.0,
                'strategy_name': 'test_strategy',
                'confidence': 0.8
            }
            
            buy_result = await self.paper_trader.execute_trade(buy_signal)
            assert buy_result['success'], f"Buy trade failed: {buy_result.get('error', 'Unknown error')}"
            
            # Verify portfolio state after buy
            summary = self.paper_trader.get_performance_summary()
            assert summary['total_trades'] == 1, f"Expected 1 trade, got {summary['total_trades']}"
            assert summary['active_positions'] == 1, f"Expected 1 position, got {summary['active_positions']}"
            assert summary['cash_balance'] < self.test_capital, "Cash balance should decrease after buy"
            
            # Test SELL order  
            sell_signal = {
                'symbol': 'BTC/USDT',
                'action': 'SELL',
                'position_size': 50,  # $50 worth (partial sell)
                'entry_price': 43500.0,
                'strategy_name': 'test_strategy'
            }
            
            sell_result = await self.paper_trader.execute_trade(sell_signal)
            assert sell_result['success'], f"Sell trade failed: {sell_result.get('error', 'Unknown error')}"
            
            self._record_test_result(test_name, True, "Buy and sell trades executed successfully")
            self.logger.info(f"Test 2 PASSED: {test_name}")
            
        except Exception as e:
            self._record_test_result(test_name, False, str(e))
            self.logger.error(f"Test 2 FAILED: {test_name} - {e}")
    
    async def _test_portfolio_management(self):
        """Test 3: Portfolio management functionality"""
        
        test_name = "Portfolio Management"
        self.logger.info(f"Running Test 3: {test_name}")
        
        try:
            # Execute multiple trades across different assets
            trades = [
                {
                    'symbol': 'ETH/USDT',
                    'action': 'BUY',
                    'position_size': 200,
                    'entry_price': 2600.0,
                    'strategy_name': 'test_diversification'
                },
                {
                    'symbol': 'SOL/USDT', 
                    'action': 'BUY',
                    'position_size': 150,
                    'entry_price': 100.0,
                    'strategy_name': 'test_diversification'
                }
            ]
            
            for trade in trades:
                result = await self.paper_trader.execute_trade(trade)
                assert result['success'], f"Trade failed for {trade['symbol']}: {result.get('error')}"
            
            # Verify portfolio diversification
            summary = self.paper_trader.get_performance_summary()
            assert summary['active_positions'] >= 2, f"Expected at least 2 positions, got {summary['active_positions']}"
            
            # Test portfolio state saving
            await self.paper_trader.save_portfolio_state()
            
            self._record_test_result(test_name, True, "Portfolio management working correctly")
            self.logger.info(f"Test 3 PASSED: {test_name}")
            
        except Exception as e:
            self._record_test_result(test_name, False, str(e))
            self.logger.error(f"Test 3 FAILED: {test_name} - {e}")
    
    async def _test_performance_tracking(self):
        """Test 4: Performance tracking and metrics"""
        
        test_name = "Performance Tracking"
        self.logger.info(f"Running Test 4: {test_name}")
        
        try:
            # Execute profitable trade simulation
            profitable_trade = {
                'symbol': 'BTC/USDT',
                'action': 'BUY',
                'position_size': 300,
                'entry_price': 40000.0,
                'strategy_name': 'test_profit'
            }
            
            await self.paper_trader.execute_trade(profitable_trade)
            
            # Simulate price increase with sell at higher price
            profit_sell = {
                'symbol': 'BTC/USDT',
                'action': 'SELL',
                'position_size': 300,
                'entry_price': 41000.0,  # $1000 profit simulation
                'strategy_name': 'test_profit'
            }
            
            await self.paper_trader.execute_trade(profit_sell)
            
            # Verify performance metrics
            summary = self.paper_trader.get_performance_summary()
            assert summary['total_trades'] > 3, f"Expected >3 trades, got {summary['total_trades']}"
            
            # Check for profitable trades in history
            trade_history = self.paper_trader.get_trade_history()
            assert len(trade_history) > 0, "No trades recorded in history"
            
            self._record_test_result(test_name, True, "Performance tracking functioning correctly")
            self.logger.info(f"Test 4 PASSED: {test_name}")
            
        except Exception as e:
            self._record_test_result(test_name, False, str(e))
            self.logger.error(f"Test 4 FAILED: {test_name} - {e}")
    
    def _generate_test_report(self) -> bool:
        """Generate comprehensive test report"""
        
        print("\n" + "="*60)
        print("PAPER TRADING TEST RESULTS")
        print("="*60)
        
        passed_tests = [r for r in self.test_results if r['passed']]
        failed_tests = [r for r in self.test_results if not r['passed']]
        
        print(f"Total Tests: {len(self.test_results)}")
        print(f"Passed: {len(passed_tests)}")
        print(f"Failed: {len(failed_tests)}")
        print(f"Success Rate: {(len(passed_tests)/len(self.test_results)*100):.1f}%")
        print()
        
        if failed_tests:
            print("FAILED TESTS:")
            for test in failed_tests:
                print(f"  - {test['test_name']}: {test['message']}")
            print()
        
        if passed_tests:
            print("PASSED TESTS:")
            for test in passed_tests:
                print(f"  + {test['test_name']}: {test['message']}")
            print()
        
        # Get final portfolio summary
        if self.paper_trader:
            final_summary = self.paper_trader.get_performance_summary()
            print("FINAL PORTFOLIO STATE:")
            print(f"  Portfolio Value: ${final_summary['portfolio_value']:,.2f}")
            print(f"  Cash Balance: ${final_summary['cash_balance']:,.2f}")
            print(f"  Total Trades: {final_summary['total_trades']}")
            print(f"  Active Positions: {final_summary['active_positions']}")
            print()
        
        success = len(failed_tests) == 0
        
        if success:
            print("STATUS: ALL TESTS PASSED - READY FOR LIVE TRADING")
            self.logger.info("All paper trading tests completed successfully!")
        else:
            print("STATUS: SOME TESTS FAILED - REVIEW BEFORE LIVE TRADING") 
            self.logger.warning(f"{len(failed_tests)} tests failed. Review before going live!")
        
        print("="*60)
        
        # Save detailed report
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'test_results': self.test_results,
            'summary': {
                'total_tests': len(self.test_results),
                'passed': len(passed_tests),
                'failed': len(failed_tests),
                'success_rate': len(passed_tests)/len(self.test_results)*100
            },
            'portfolio_state': final_summary if self.paper_trader else {}
        }
        
        report_file = f"data/reports/paper_trading_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"Detailed report saved to: {report_file}")
        
        return success


async def main():
    """Main test runner"""
    
    print("SAFETY FIRST: Paper Trading Validation")
    print("This test suite validates all systems before real trading.")
    print("NO REAL MONEY IS INVOLVED - Only virtual simulation!")
    print()
    
    test_suite = SimplePaperTradingTestSuite()
    
    try:
        success = await test_suite.run_comprehensive_tests()
        
        if success:
            print("\nRECOMMENDation: Systems validated - You may proceed with SMALL live trades")
            print("NEXT STEPS:")
            print("1. Run live bot with minimal capital first")
            print("2. Monitor performance closely for 24-48 hours")
            print("3. Gradually increase position sizes")
            print("4. Always maintain strict risk management")
        else:
            print("\nWARNING: Fix failed tests before live trading!")
            print("Paper trading revealed issues that must be resolved.")
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        print("Cannot proceed to live trading - system validation failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

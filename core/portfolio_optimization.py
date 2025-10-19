#!/usr/bin/env python3
"""
🎯 ADVANCED PORTFOLIO OPTIMIZATION SYSTEM
Modern Portfolio Theory, Black-Litterman Model & Factor Analysis
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Advanced optimization imports
try:
    from sklearn.decomposition import PCA, FactorAnalysis
    from sklearn.covariance import LedoitWolf, EmpiricalCovariance
    from sklearn.preprocessing import StandardScaler
    import cvxpy as cp
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    print("⚠️ Install scipy, scikit-learn, cvxpy for full portfolio optimization")

@dataclass
class AssetMetrics:
    """Individual asset performance metrics"""
    symbol: str
    expected_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    beta: float
    alpha: float
    correlation_with_market: float
    
@dataclass
class PortfolioAllocation:
    """Portfolio allocation result"""
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    var_95: float  # Value at Risk
    cvar_95: float  # Conditional Value at Risk
    maximum_drawdown: float
    diversification_ratio: float

class ModernPortfolioTheory:
    """Advanced Modern Portfolio Theory implementation"""
    
    def __init__(self):
        self.returns_data = pd.DataFrame()
        self.covariance_matrix = None
        self.expected_returns = None
        self.risk_free_rate = 0.02  # 2% risk-free rate
        self.lookback_days = 252  # 1 year of data
        
    def add_price_data(self, symbol: str, prices: List[float], timestamps: List[datetime]):
        """Add price data for an asset"""
        df = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices
        })
        df.set_index('timestamp', inplace=True)
        
        # Calculate returns
        returns = df['price'].pct_change().dropna()
        self.returns_data[symbol] = returns
        
        print(f"📊 Added {symbol}: {len(returns)} return observations")
    
    def calculate_asset_metrics(self, symbol: str) -> AssetMetrics:
        """Calculate comprehensive metrics for an asset"""
        if symbol not in self.returns_data.columns:
            raise ValueError(f"No data for {symbol}")
        
        returns = self.returns_data[symbol].dropna()
        
        # Basic metrics
        expected_return = returns.mean() * 252  # Annualized
        volatility = returns.std() * np.sqrt(252)  # Annualized
        sharpe_ratio = (expected_return - self.risk_free_rate) / volatility
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # Market relationship (using equal-weighted portfolio as market proxy)
        if len(self.returns_data.columns) > 1:
            market_returns = self.returns_data.mean(axis=1)
            beta = np.cov(returns, market_returns)[0, 1] / np.var(market_returns)
            alpha = expected_return - (self.risk_free_rate + beta * (market_returns.mean() * 252 - self.risk_free_rate))
            correlation = np.corrcoef(returns, market_returns)[0, 1]
        else:
            beta, alpha, correlation = 1.0, 0.0, 1.0
        
        return AssetMetrics(
            symbol=symbol,
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            beta=beta,
            alpha=alpha,
            correlation_with_market=correlation
        )
    
    def estimate_expected_returns(self, method: str = 'historical') -> np.ndarray:
        """Estimate expected returns using various methods"""
        if method == 'historical':
            # Simple historical mean
            self.expected_returns = self.returns_data.mean().values * 252
            
        elif method == 'ewm':
            # Exponentially weighted moving average
            ewm_returns = self.returns_data.ewm(halflife=63).mean().iloc[-1].values * 252
            self.expected_returns = ewm_returns
            
        elif method == 'capm':
            # Capital Asset Pricing Model
            market_returns = self.returns_data.mean(axis=1) * 252
            market_return = market_returns.mean()
            
            expected_returns = []
            for symbol in self.returns_data.columns:
                asset_returns = self.returns_data[symbol]
                beta = np.cov(asset_returns, market_returns)[0, 1] / np.var(market_returns)
                expected_return = self.risk_free_rate + beta * (market_return - self.risk_free_rate)
                expected_returns.append(expected_return)
            
            self.expected_returns = np.array(expected_returns)
        
        return self.expected_returns
    
    def estimate_covariance_matrix(self, method: str = 'ledoit_wolf') -> np.ndarray:
        """Estimate covariance matrix using advanced techniques"""
        if method == 'sample':
            # Sample covariance
            self.covariance_matrix = self.returns_data.cov().values * 252
            
        elif method == 'ledoit_wolf':
            # Ledoit-Wolf shrinkage estimator
            if OPTIMIZATION_AVAILABLE:
                lw = LedoitWolf()
                self.covariance_matrix = lw.fit(self.returns_data.fillna(0)).covariance_ * 252
            else:
                self.covariance_matrix = self.returns_data.cov().values * 252
                
        elif method == 'exponential':
            # Exponentially weighted covariance
            ewm_cov = self.returns_data.ewm(halflife=63).cov().iloc[-len(self.returns_data.columns):].values * 252
            self.covariance_matrix = ewm_cov
        
        return self.covariance_matrix
    
    def optimize_portfolio(self, 
                          objective: str = 'max_sharpe',
                          constraints: Dict = None,
                          expected_returns: np.ndarray = None,
                          covariance_matrix: np.ndarray = None) -> PortfolioAllocation:
        """Optimize portfolio using various objectives"""
        
        if expected_returns is None:
            expected_returns = self.estimate_expected_returns()
        if covariance_matrix is None:
            covariance_matrix = self.estimate_covariance_matrix()
        
        n_assets = len(expected_returns)
        
        # Default constraints
        default_constraints = {
            'min_weight': 0.0,
            'max_weight': 1.0,
            'max_concentration': 0.4,  # No single asset > 40%
            'min_positions': 3,
            'target_return': None,
            'target_volatility': None
        }
        
        if constraints:
            default_constraints.update(constraints)
        
        constraints_obj = default_constraints
        
        if objective == 'max_sharpe':
            weights = self._optimize_max_sharpe(expected_returns, covariance_matrix, constraints_obj)
        elif objective == 'min_variance':
            weights = self._optimize_min_variance(covariance_matrix, constraints_obj)
        elif objective == 'max_return':
            weights = self._optimize_max_return(expected_returns, constraints_obj)
        elif objective == 'risk_parity':
            weights = self._optimize_risk_parity(covariance_matrix, constraints_obj)
        elif objective == 'target_return':
            weights = self._optimize_target_return(expected_returns, covariance_matrix, constraints_obj)
        else:
            raise ValueError(f"Unknown objective: {objective}")
        
        return self._create_portfolio_allocation(weights, expected_returns, covariance_matrix)
    
    def _optimize_max_sharpe(self, expected_returns: np.ndarray, cov_matrix: np.ndarray, constraints: Dict) -> np.ndarray:
        """Optimize for maximum Sharpe ratio"""
        n_assets = len(expected_returns)
        
        def neg_sharpe(weights):
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            portfolio_std = np.sqrt(portfolio_variance)
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_std
            return -sharpe
        
        # Constraints
        cons = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},  # Weights sum to 1
        ]
        
        # Individual weight constraints
        bounds = [(constraints['min_weight'], min(constraints['max_weight'], constraints['max_concentration'])) 
                 for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        x0 = np.array([1/n_assets] * n_assets)
        
        result = minimize(neg_sharpe, x0, method='SLSQP', bounds=bounds, constraints=cons)
        
        if result.success:
            return result.x
        else:
            print(f"⚠️ Optimization failed: {result.message}")
            return x0
    
    def _optimize_min_variance(self, cov_matrix: np.ndarray, constraints: Dict) -> np.ndarray:
        """Optimize for minimum variance"""
        n_assets = cov_matrix.shape[0]
        
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        bounds = [(constraints['min_weight'], constraints['max_weight']) for _ in range(n_assets)]
        x0 = np.array([1/n_assets] * n_assets)
        
        result = minimize(portfolio_variance, x0, method='SLSQP', bounds=bounds, constraints=cons)
        return result.x if result.success else x0
    
    def _optimize_max_return(self, expected_returns: np.ndarray, constraints: Dict) -> np.ndarray:
        """Optimize for maximum expected return"""
        n_assets = len(expected_returns)
        
        def neg_return(weights):
            return -np.sum(weights * expected_returns)
        
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        bounds = [(constraints['min_weight'], constraints['max_weight']) for _ in range(n_assets)]
        x0 = np.array([1/n_assets] * n_assets)
        
        result = minimize(neg_return, x0, method='SLSQP', bounds=bounds, constraints=cons)
        return result.x if result.success else x0
    
    def _optimize_risk_parity(self, cov_matrix: np.ndarray, constraints: Dict) -> np.ndarray:
        """Risk parity optimization (equal risk contribution)"""
        n_assets = cov_matrix.shape[0]
        
        def risk_parity_objective(weights):
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            marginal_contrib = np.dot(cov_matrix, weights)
            contrib = weights * marginal_contrib / portfolio_variance
            target_contrib = 1.0 / n_assets
            return np.sum((contrib - target_contrib) ** 2)
        
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        bounds = [(0.01, constraints['max_weight']) for _ in range(n_assets)]  # Minimum 1% to avoid division by zero
        x0 = np.array([1/n_assets] * n_assets)
        
        result = minimize(risk_parity_objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
        return result.x if result.success else x0
    
    def _optimize_target_return(self, expected_returns: np.ndarray, cov_matrix: np.ndarray, constraints: Dict) -> np.ndarray:
        """Optimize for target return with minimum variance"""
        target_return = constraints.get('target_return', expected_returns.mean())
        n_assets = len(expected_returns)
        
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        cons = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
            {'type': 'eq', 'fun': lambda x: np.sum(x * expected_returns) - target_return}
        ]
        
        bounds = [(constraints['min_weight'], constraints['max_weight']) for _ in range(n_assets)]
        x0 = np.array([1/n_assets] * n_assets)
        
        result = minimize(portfolio_variance, x0, method='SLSQP', bounds=bounds, constraints=cons)
        return result.x if result.success else x0
    
    def _create_portfolio_allocation(self, weights: np.ndarray, expected_returns: np.ndarray, cov_matrix: np.ndarray) -> PortfolioAllocation:
        """Create PortfolioAllocation object from optimization results"""
        
        # Portfolio metrics
        portfolio_return = np.sum(weights * expected_returns)
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        # Risk metrics
        var_95 = portfolio_return - 1.645 * portfolio_volatility  # 95% VaR
        cvar_95 = portfolio_return - 2.33 * portfolio_volatility   # 95% CVaR (approximation)
        
        # Diversification ratio
        weighted_avg_vol = np.sum(weights * np.sqrt(np.diag(cov_matrix)))
        diversification_ratio = weighted_avg_vol / portfolio_volatility
        
        # Weight dictionary
        weight_dict = {symbol: float(weight) for symbol, weight in zip(self.returns_data.columns, weights)}
        
        return PortfolioAllocation(
            weights=weight_dict,
            expected_return=portfolio_return,
            expected_volatility=portfolio_volatility,
            sharpe_ratio=sharpe_ratio,
            var_95=var_95,
            cvar_95=cvar_95,
            maximum_drawdown=0.0,  # Would need simulation to calculate properly
            diversification_ratio=diversification_ratio
        )

class BlackLittermanModel:
    """Advanced Black-Litterman portfolio optimization"""
    
    def __init__(self, risk_aversion: float = 3.0):
        self.risk_aversion = risk_aversion
        self.market_caps = {}
        self.views = []
        self.view_uncertainty = []
        
    def set_market_caps(self, market_caps: Dict[str, float]):
        """Set market capitalizations for assets"""
        self.market_caps = market_caps
    
    def add_view(self, assets: List[str], weights: List[float], expected_return: float, confidence: float):
        """Add a view (opinion) about expected returns
        
        Args:
            assets: List of asset symbols
            weights: Weights for the view (should sum to 1 for absolute views, 0 for relative)
            expected_return: Expected return for this view
            confidence: Confidence level (0-1, higher = more confident)
        """
        view = {
            'assets': assets,
            'weights': weights,
            'expected_return': expected_return,
            'confidence': confidence
        }
        self.views.append(view)
        
        # Convert confidence to uncertainty (lower confidence = higher uncertainty)
        uncertainty = (1 - confidence) * 0.1  # Scale factor
        self.view_uncertainty.append(uncertainty)
    
    def calculate_implied_returns(self, cov_matrix: np.ndarray, market_weights: np.ndarray) -> np.ndarray:
        """Calculate implied equilibrium returns from market capitalization"""
        # Π = δΣw (where δ is risk aversion, Σ is covariance matrix, w is market cap weights)
        implied_returns = self.risk_aversion * np.dot(cov_matrix, market_weights)
        return implied_returns
    
    def optimize_with_views(self, 
                           symbols: List[str], 
                           cov_matrix: np.ndarray, 
                           market_weights: np.ndarray = None,
                           tau: float = 0.025) -> Tuple[np.ndarray, np.ndarray]:
        """Perform Black-Litterman optimization with views"""
        
        n_assets = len(symbols)
        
        # Use equal weights if no market cap data
        if market_weights is None:
            market_weights = np.array([1/n_assets] * n_assets)
        
        # Calculate implied equilibrium returns
        pi = self.calculate_implied_returns(cov_matrix, market_weights)
        
        if not self.views:
            # No views, return equilibrium
            return pi, market_weights
        
        # Construct view matrices
        P = np.zeros((len(self.views), n_assets))  # Picking matrix
        Q = np.zeros(len(self.views))  # Views vector
        
        for i, view in enumerate(self.views):
            Q[i] = view['expected_return']
            for j, asset in enumerate(view['assets']):
                if asset in symbols:
                    asset_idx = symbols.index(asset)
                    P[i, asset_idx] = view['weights'][j]
        
        # View uncertainty matrix (diagonal)
        Omega = np.diag(self.view_uncertainty)
        
        # Black-Litterman formula
        # μ_BL = [(τΣ)^-1 + P'Ω^-1P]^-1 [(τΣ)^-1 π + P'Ω^-1 Q]
        
        tau_sigma = tau * cov_matrix
        tau_sigma_inv = np.linalg.inv(tau_sigma)
        
        # Middle term
        middle = np.linalg.inv(tau_sigma_inv + P.T @ np.linalg.inv(Omega) @ P)
        
        # Right side
        right_side = tau_sigma_inv @ pi + P.T @ np.linalg.inv(Omega) @ Q
        
        # New expected returns
        mu_bl = middle @ right_side
        
        # New covariance matrix
        sigma_bl = middle
        
        # Optimize portfolio with BL inputs
        def neg_utility(weights):
            portfolio_return = np.sum(weights * mu_bl)
            portfolio_variance = np.dot(weights.T, np.dot(sigma_bl, weights))
            utility = portfolio_return - 0.5 * self.risk_aversion * portfolio_variance
            return -utility
        
        from scipy.optimize import minimize
        
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        bounds = [(0.0, 1.0) for _ in range(n_assets)]
        x0 = market_weights
        
        result = minimize(neg_utility, x0, method='SLSQP', bounds=bounds, constraints=cons)
        
        optimal_weights = result.x if result.success else market_weights
        
        return mu_bl, optimal_weights

class FactorModel:
    """Multi-factor risk model for portfolio attribution"""
    
    def __init__(self):
        self.factors = ['Market', 'Size', 'Value', 'Momentum', 'Quality', 'Volatility']
        self.factor_loadings = pd.DataFrame()
        self.factor_returns = pd.DataFrame()
        
    def estimate_factor_loadings(self, returns_data: pd.DataFrame, market_returns: pd.Series = None):
        """Estimate factor loadings using regression analysis"""
        
        if market_returns is None:
            market_returns = returns_data.mean(axis=1)
        
        factor_data = pd.DataFrame(index=returns_data.index)
        
        # Market factor
        factor_data['Market'] = market_returns
        
        # Size factor (SMB - Small Minus Big)
        # Simplified: use volatility as proxy for size
        volatilities = returns_data.rolling(20).std()
        high_vol = volatilities.quantile(0.7, axis=1)
        low_vol = volatilities.quantile(0.3, axis=1)
        factor_data['Size'] = high_vol - low_vol
        
        # Value factor (HML - High Minus Low)
        # Simplified: use momentum as proxy
        momentum = returns_data.rolling(20).mean()
        high_mom = momentum.quantile(0.7, axis=1)
        low_mom = momentum.quantile(0.3, axis=1)
        factor_data['Value'] = low_mom - high_mom  # Value is opposite of momentum
        
        # Momentum factor
        factor_data['Momentum'] = high_mom - low_mom
        
        # Quality factor (profitability proxy using Sharpe ratios)
        sharpe_ratios = returns_data.rolling(60).mean() / returns_data.rolling(60).std()
        factor_data['Quality'] = sharpe_ratios.mean(axis=1)
        
        # Volatility factor
        factor_data['Volatility'] = returns_data.rolling(20).std().mean(axis=1)
        
        self.factor_returns = factor_data.fillna(0)
        
        # Estimate loadings for each asset
        loadings = {}
        for symbol in returns_data.columns:
            asset_returns = returns_data[symbol].dropna()
            factor_subset = self.factor_returns.loc[asset_returns.index]
            
            # Multiple regression
            from sklearn.linear_model import LinearRegression
            if OPTIMIZATION_AVAILABLE:
                reg = LinearRegression()
                reg.fit(factor_subset.fillna(0), asset_returns)
                loadings[symbol] = reg.coef_
            else:
                # Simple correlation-based loadings
                correlations = []
                for factor in self.factors:
                    corr = asset_returns.corr(factor_subset[factor])
                    correlations.append(corr if not np.isnan(corr) else 0)
                loadings[symbol] = correlations
        
        self.factor_loadings = pd.DataFrame(loadings, index=self.factors).T
        return self.factor_loadings
    
    def calculate_risk_attribution(self, portfolio_weights: Dict[str, float]) -> Dict:
        """Calculate risk attribution by factor"""
        
        if self.factor_loadings.empty:
            return {}
        
        # Portfolio factor loadings
        portfolio_loadings = {}
        for factor in self.factors:
            loading = 0
            for symbol, weight in portfolio_weights.items():
                if symbol in self.factor_loadings.index:
                    loading += weight * self.factor_loadings.loc[symbol, factor]
            portfolio_loadings[factor] = loading
        
        # Factor risk contributions (simplified)
        factor_variances = self.factor_returns.var()
        total_risk = sum(portfolio_loadings[factor]**2 * factor_variances.get(factor, 0) 
                        for factor in self.factors)
        
        risk_attribution = {}
        for factor in self.factors:
            factor_risk = portfolio_loadings[factor]**2 * factor_variances.get(factor, 0)
            risk_attribution[factor] = {
                'loading': portfolio_loadings[factor],
                'risk_contribution': factor_risk / total_risk if total_risk > 0 else 0,
                'risk_contribution_pct': (factor_risk / total_risk * 100) if total_risk > 0 else 0
            }
        
        return risk_attribution

# Global instances
portfolio_optimizer = ModernPortfolioTheory()
black_litterman = BlackLittermanModel()
factor_model = FactorModel()

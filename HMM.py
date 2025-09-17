import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import warnings
from src.base.indicatorManager import AllIndicators
from src.utils.dbAccessor import DBAccessor
import joblib
import os
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

db_accessor = DBAccessor()
db_accessor.db_path = "NSE_Weekly.db"

db_accessor.connect()
db_accessor.set_columns(columns=['date', 'open', 'high', 'low', 'close', 'volume'])
db_accessor.set_cdl_interval(interval='1D')

BTC_data: pd.DataFrame = db_accessor.fetch_data(symbol="INDUSINDBK")

class MarketRegimeHMM:
    def __init__(self, n_components=4, covariance_type="full", n_iter=1000, scaling_method='robust',
                 init_params='stmc', params='stmc', random_state=42, tol=1e-5, min_covar=1e-3,
                 init_method='kmeans', n_init=1, verbose=False):
        """
        Initialize HMM for market regime detection
        
        Args:
            n_components: Number of hidden states (Bull, Bear, Range, Chop)
                # Possible values: 2-6 (recommended: 3-5)
                # More components = more complex model, but may overfit
            
            covariance_type: Type of covariance parameters
                # Possible values:
                # - 'full': Full covariance matrix (most flexible, more parameters)
                # - 'tied': All states share the same covariance matrix (more constrained)
                # - 'diag': Diagonal covariance matrix (assumes features are independent)
                # - 'spherical': Spherical covariance matrix (most constrained)
            
            n_iter: Maximum number of iterations for EM algorithm
                # Possible values: 100-2000
                # Higher values = more training time, but may find better solution
            
            scaling_method: Method for scaling features
                # Possible values:
                # - 'robust': Less sensitive to outliers (recommended for financial data)
                # - 'standard': Standard scaling (mean=0, std=1)
            
            init_params: Parameters to initialize
                # Possible values: combination of:
                # - 's': Start probabilities
                # - 't': Transition probabilities
                # - 'm': Means
                # - 'c': Covariances
                # Example: 'stmc' initializes all parameters
            
            params: Parameters to update during training
                # Same possible values as init_params
                # Controls which parameters are updated during EM algorithm
            
            random_state: Random seed for reproducibility
                # Any integer value
                # None for random initialization
            
            tol: Convergence threshold
                # Possible values: 1e-2 to 1e-7
                # Lower values = more precise convergence, but longer training
            
            min_covar: Minimum covariance value (regularization)
                # Possible values: 1e-6 to 1e-2
                # Higher values = more regularization, prevents overfitting
                # - 1e-6: Minimal regularization
                # - 1e-3: Moderate regularization (default)
                # - 1e-2: Strong regularization
            
            init_method: Method for initialization
                # Possible values:
                # - 'kmeans': K-means clustering (default, most stable)
                # - 'random': Random initialization
                # - 'user': User-provided initialization
            
            n_init: Number of initializations to try
                # Possible values: 1-20
                # Higher values = better chance of finding global optimum, but slower
            
            verbose: Whether to print progress during training
                # Possible values: True/False
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.scaling_method = scaling_method
        self.init_params = init_params
        self.params = params
        self.random_state = random_state
        self.tol = tol
        self.min_covar = min_covar
        self.init_method = init_method
        self.n_init = n_init
        self.verbose = verbose
        
        # Initialize HMM model with specified parameters
        self.model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
            tol=tol,
            init_params=init_params,
            params=params,
            verbose=verbose,
            min_covar=min_covar  # Add regularization parameter
        )
        
        self.scalers = {}  # Dictionary to store feature-specific scalers
        self.is_fitted = False
        self.regime_labels = ['Bull', 'Bear', 'Range', 'Chop']
        self.feature_names = []
        
    def _initialize_scalers(self):
        """Initialize appropriate scalers for each feature type"""
        # Features that should use robust scaling (less sensitive to outliers)
        robust_features = ['ema_diff_pct', 'atr', 'atr_change', 'log_return']
        
        # Features that should use standard scaling
        standard_features = ['rsi']
        
        for feature in robust_features:
            self.scalers[feature] = RobustScaler() if self.scaling_method == 'robust' else StandardScaler()
            
        for feature in standard_features:
            self.scalers[feature] = StandardScaler()
    
    def _scale_features(self, features, is_training=False):
        """
        Scale features using appropriate scalers for each feature type
        
        Args:
            features: DataFrame with features to scale
            is_training: Whether this is training data (to fit scalers) or prediction data
        """
        if not self.scalers:
            self._initialize_scalers()
            
        scaled_features = pd.DataFrame(index=features.index)
        
        for feature in features.columns:
            if feature in self.scalers:
                if is_training:
                    scaled_features[feature] = self.scalers[feature].fit_transform(features[[feature]])
                else:
                    scaled_features[feature] = self.scalers[feature].transform(features[[feature]])
            else:
                # If no specific scaler is defined, use the default method
                scaler = RobustScaler() if self.scaling_method == 'robust' else StandardScaler()
                if is_training:
                    scaled_features[feature] = scaler.fit_transform(features[[feature]])
                else:
                    scaled_features[feature] = scaler.transform(features[[feature]])
                self.scalers[feature] = scaler
                
        return features
    
    def calculate_features(self, df):
        """
        Calculate technical features based on your existing approach
        
        Args:
            df: DataFrame with OHLCV data
        Returns:
            DataFrame with calculated features
        """
        features = pd.DataFrame(index=df.index)

        # Calculate EMAs
        df['EMA_20'] = df['Open'].ewm(span=20, min_periods=20).mean()
        df['EMA_50'] = df['Open'].ewm(span=50, min_periods=50).mean()
        
        # EMA difference percentage
        features['ema_diff_pct'] = (df['EMA_20'] - df['EMA_50']) / df['EMA_50'] * 100
        
        # RSI with smoothing
        features['rsi'] = self.calculate_rsi(df['Open']).rolling(window=5).mean()
        
        # Log return standard deviation (20-day)
        log_returns = np.log(df['Open'] / df['Open'].shift(1))
        features['log_return_std'] = log_returns.rolling(window=20).std()
        
        # Log returns
        features['log_return'] = log_returns
        
        # Momentum (10-day)
        features['momentum10'] = df['Open'].pct_change(10) * 100
        
        # Remove NaN values
        features = features.dropna()
        self.feature_names = features.columns.tolist()
        
        return features
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_atr(self, df, period=20):
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Open'].shift())
        low_close = np.abs(df['Low'] - df['Open'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(period).mean()
        return atr
    
    def prepare_data(self, features, is_training=False):
        """
        Prepare and scale features for HMM
        
        Args:
            features: DataFrame with features to prepare
            is_training: Whether this is training data or prediction data
        """
        # Remove any remaining NaN values
        features_clean = features.dropna()
        
        # Scale features using appropriate scalers
        features_scaled = self._scale_features(features_clean, is_training)
        
        return features_scaled, features_clean.index
    
    def fit(self, df):
        """
        Fit the HMM model to the data
        
        Args:
            df: DataFrame with OHLCV data
        """
        # Calculate features
        features = self.calculate_features(df)
        
        # Prepare data with training flag
        X, self.feature_index = self.prepare_data(features, is_training=True)
        
        # Fit HMM
        self.model.fit(X)
        
        # Predict states
        self.states = self.model.predict(X)
        
        # Analyze states to assign regime labels
        self._assign_regime_labels(X)
        
        self.is_fitted = True
        
        return self
    
    def _assign_regime_labels(self, X):
        """
        Keep states as pure hidden states without regime labels
        """
        self.state_to_regime = {i: f'State_{i}' for i in range(self.n_components)}
    
    def predict(self, df):
        """
        Predict regimes for new data
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        features = self.calculate_features(df)
        X, _ = self.prepare_data(features, is_training=False)
        
        states = self.model.predict(X)
        regimes = [self.state_to_regime[state] for state in states]
        
        return regimes, states, features
    
    def get_regime_probabilities(self, df):
        """
        Get probability of each regime for each time period
        """
        features = self.calculate_features(df)
        X, _ = self.prepare_data(features)
        
        state_probs = self.model.predict_proba(X)
        
        # Convert to regime probabilities
        regime_probs = pd.DataFrame(
            state_probs, 
            columns=[f'State_{i}' for i in range(self.n_components)],
            index=self.feature_index
        )
        
        # Add regime labels
        for state, regime in self.state_to_regime.items():
            regime_probs[f'{regime}_prob'] = regime_probs[f'State_{state}']
        
        return regime_probs
    
    def plot_regimes(self, df, price_col='Close'):
        """
        Plot price data with detected states using Plotly
        """
        regimes, states, features_df = self.predict(df)
        
        # Create a copy of df with consistent index
        df_plot = df.copy()
        df_plot = df_plot.loc[features_df.index]
        
        # Create color map for states
        state_colors = {
            'State_0': 'rgba(39, 174, 96, 0.30)',     # Bright green
            'State_1': 'rgba(192, 57, 43, 0.30)',     # Vivid red
            'State_2': 'rgba(41, 128, 185, 0.30)',   # Deep blue
            'State_3': 'rgba(142, 68, 173, 0.30)',     # Strong purple
            'State_4': 'rgba(255, 165, 0, 0.30)'      # Vibrant orange
        }
        
        # Create state label column
        df_plot['state'] = [f'State_{s}' for s in states]
        
        # Identify state change points
        df_plot['state_change'] = (df_plot['state'] != df_plot['state'].shift(1)).cumsum()
        
        # Create plot structure
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            specs=[[{"secondary_y": True}], [{}]]
        )
        
        # Add price traces
        fig.add_trace(go.Candlestick(
            x=df_plot.index,
            open=df_plot['Open'],
            high=df_plot['High'],
            low=df_plot['Low'],
            close=df_plot['Close'],
            name='OHLC',
            increasing_line_color='#2ecc71',
            decreasing_line_color='#e74c3c'
        ), row=1, col=1)
        
        # Add EMA traces
        fig.add_trace(go.Scatter(
            x=df_plot.index,
            y=df_plot['EMA_20'],
            line=dict(color='#3498db', width=1.5),
            name='EMA 20'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df_plot.index,
            y=df_plot['EMA_50'],
            line=dict(color='#f39c12', width=1.5),
            name='EMA 50'
        ), row=1, col=1)
        
        # Add state backgrounds
        for group, data in df_plot.groupby('state_change'):
            state_type = data['state'].iloc[0]
            fig.add_vrect(
                x0=data.index[0],
                x1=data.index[-1],
                fillcolor=state_colors[state_type],
                layer="below",
                line_width=0,
                row=1, col=1
            )
        
        # Add RSI
        fig.add_trace(go.Scatter(
            x=df_plot.index,
            y=features_df['rsi'],
            line=dict(color='#9b59b6', width=1.5),
            name='RSI'
        ), row=2, col=1)
        
        fig.add_hline(y=30, line_dash="dash", line_color="#95a5a6", row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#95a5a6", row=2, col=1)
        
        # Layout configuration
        fig.update_layout(
            title='Market State Analysis',
            height=800,
            template='plotly_dark',
            hovermode='x unified',
            legend=dict(orientation='h', y=1.02, x=0),
            xaxis_rangeslider_visible=False
        )
        
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="RSI", range=[0,100], row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        
        # Add custom legend for states
        for state, color in state_colors.items():
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(size=10, color=color),
                name=state,
                hoverinfo='none'
            ))
        
        return fig
    
    def save_results(self, df, output_dir='hmm_results'):
        """
        Save HMM results to CSV files and plot to HTML
        
        Args:
            df: DataFrame with price data
            output_dir: Directory to save results
        """        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get current timestamp for file names
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Get regime predictions and features
        regimes, states, features_df = self.predict(df)
        
        # Create results DataFrame with consistent index
        results_df = pd.DataFrame({
            'date': features_df.index,
            'regime': regimes,
            'state': states
        })
        
        # Add regime probabilities
        regime_probs = self.get_regime_probabilities(df)
        results_df = pd.concat([results_df, regime_probs], axis=1)
        
        # Save results to CSV
        results_file = os.path.join(output_dir, f'hmm_results_{timestamp}.csv')
        results_df.to_csv(results_file)
        
        # Save regime statistics (both sequential and aggregated)
        stats_dict = self.get_regime_statistics(df)
        sequential_stats_file = os.path.join(output_dir, f'hmm_sequential_statistics_{timestamp}.csv')
        aggregated_stats_file = os.path.join(output_dir, f'hmm_aggregated_statistics_{timestamp}.csv')
        stats_dict['sequential'].to_csv(sequential_stats_file)
        stats_dict['aggregated'].to_csv(aggregated_stats_file)
        
        # Save transition matrix
        trans_matrix = self.get_transition_matrix()
        trans_file = os.path.join(output_dir, f'hmm_transitions_{timestamp}.csv')
        trans_matrix.to_csv(trans_file)
        
        # Create and save plot using the same data points
        fig = self.plot_regimes(df)
        plot_file = os.path.join(output_dir, f'hmm_plot_{timestamp}.html')
        fig.write_html(plot_file)
        
        print(f"Results saved to {output_dir}:")
        print(f"- Regime predictions: {results_file}")
        print(f"- Sequential statistics: {sequential_stats_file}")
        print(f"- Aggregated statistics: {aggregated_stats_file}")
        print(f"- Transition matrix: {trans_file}")
        print(f"- Interactive plot: {plot_file}")
        
        return results_file, sequential_stats_file, aggregated_stats_file, trans_file, plot_file
    
    def get_transition_matrix(self):
        """
        Get the transition probability matrix
        """
        return pd.DataFrame(
            self.model.transmat_,
            columns=[f'To_{self.state_to_regime[i]}' for i in range(self.n_components)],
            index=[f'From_{self.state_to_regime[i]}' for i in range(self.n_components)]
        )
    
    def get_regime_statistics(self, df):
        """
        Get statistics for each state, both sequential and aggregated
        """
        _, states, features_df = self.predict(df)
        
        # Calculate features and get the correct index
        features = self.calculate_features(df)
        X, feature_index = self.prepare_data(features)
        
        # Calculate returns using the feature index
        returns = df.loc[feature_index, 'Open'].pct_change().dropna()
        
        # Calculate sequential statistics
        sequential_stats = self._calculate_sequential_statistics(states, returns, df)
        
        # Calculate aggregated statistics
        aggregated_stats = {}
        
        # Group sequential stats by state
        state_sequential_stats = {}
        for period_id, stats in sequential_stats.items():
            state = int(period_id.split('_')[1])  # Extract state number
            if state not in state_sequential_stats:
                state_sequential_stats[state] = []
            state_sequential_stats[state].append(stats)
        
        for state in range(self.n_components):
            state_mask = np.array(states) == state
            state_returns = returns[state_mask[1:]]  # Align with returns
            
            # Get feature statistics for this state
            state_features = features_df.loc[feature_index[state_mask]]
            feature_stats = {}
            
            # Calculate statistics for each feature
            for feature in state_features.columns:
                feature_stats[f'{feature}_mean'] = state_features[feature].mean()
                feature_stats[f'{feature}_std'] = state_features[feature].std()
            
            # Calculate mean simulated returns and CAGR from sequential stats
            sim_ret = 0.0
            cagr = 0.0
            if state in state_sequential_stats:
                sim_returns = [stats['simulated_return'] for stats in state_sequential_stats[state]]
                cagrs = [stats['cagr'] for stats in state_sequential_stats[state]]
                sim_ret = np.mean(sim_returns) if sim_returns else 0.0
                cagr = np.mean(cagrs) if cagrs else 0.0
            
            sharpe = state_returns.mean() / state_returns.std() if state_returns.std() > 0 else 0

            cumulative = (1 + state_returns).cumprod()
            drawdown = (cumulative / cumulative.cummax()) - 1
            max_dd = drawdown.min()
            
            aggregated_stats[f'State_{state}'] = {
                'count': len(state_returns),
                'mean_return': state_returns.mean(),
                'volatility': state_returns.std(),
                'sharpe_ratio': sharpe * np.sqrt(252),
                'max_drawdown': max_dd,
                'mean_simulated_return': sim_ret,
                'mean_cagr': cagr,
                'avg_volume': df.loc[feature_index[state_mask], 'Volume'].mean() if 'Volume' in df.columns else None,
                **feature_stats  # Add all feature statistics
            }
        
        return {
            'sequential': pd.DataFrame(sequential_stats).T,
            'aggregated': pd.DataFrame(aggregated_stats).T
        }
    
    def _calculate_sequential_statistics(self, states, returns, df):
        """
        Calculate statistics for each sequential occurrence of states
        """
        # Convert states to numpy array for easier manipulation
        states_array = np.array(states)
        
        # Find state change points
        state_changes = np.where(states_array[:-1] != states_array[1:])[0] + 1
        state_changes = np.concatenate(([0], state_changes, [len(states_array)]))
        
        # Calculate features to get the correct index
        features = self.calculate_features(df)
        _, feature_index = self.prepare_data(features)
        
        sequential_stats = {}
        state_count = {i: 0 for i in range(self.n_components)}
        
        # Calculate statistics for each state period
        for i in range(len(state_changes) - 1):
            start_idx = state_changes[i]
            end_idx = state_changes[i + 1]
            current_state = states_array[start_idx]
            
            # Get returns for this period
            period_returns = returns[start_idx:end_idx]
            
            # Get features for this period
            period_features = features.loc[feature_index[start_idx:end_idx]]
            feature_stats = {}
            
            # Calculate statistics for each feature
            for feature in period_features.columns:
                feature_stats[f'{feature}_mean'] = period_features[feature].mean()
                feature_stats[f'{feature}_std'] = period_features[feature].std()
            
            # Create state period identifier
            state_count[current_state] += 1
            state_period_id = f'State_{current_state}_Period_{state_count[current_state]}'

            # Calculate simulated returns and CAGR for this period
            period_data = df.loc[feature_index[start_idx:end_idx+1]]
            
            # Initialize default values
            sim_ret = 0.0
            cagr = 0.0
            
            # Only calculate if we have data
            if len(period_data) > 0:
                try:
                    entry = period_data['Open'].iloc[0]
                    exit = period_data['Open'].iloc[-1]
                    days = (period_data.index[-1] - period_data.index[0]).days
                    sim_ret = (exit - entry) / entry
                    cagr = ((exit / entry) ** (365 / days)) - 1 if days > 0 else 0
                except (IndexError, KeyError):
                    # If any error occurs in calculation, keep default values
                    pass

            sharpe = period_returns.mean() / period_returns.std() if period_returns.std() > 0 else 0

            cumulative = (1 + period_returns).cumprod()
            drawdown = (cumulative / cumulative.cummax()) - 1
            max_dd = drawdown.min()
            
            # Calculate statistics for this period
            sequential_stats[state_period_id] = {
                'start_date': feature_index[start_idx],
                'end_date': feature_index[end_idx-1],
                'duration': end_idx - start_idx,
                'mean_return': period_returns.mean(),
                'volatility': period_returns.std(),
                'sharpe_ratio': sharpe * np.sqrt(252),
                'max_drawdown': max_dd,
                'simulated_return': sim_ret,
                'cagr': cagr,
                'avg_volume': df.loc[feature_index[start_idx:end_idx], 'Volume'].mean() if 'Volume' in df.columns else None,
                **feature_stats  # Add all feature statistics
            }
        
        return sequential_stats

    def walk_forward_test(self, df, min_train_years=3, max_train_years=7, test_years=1, output_dir='walk_forward_results'):
        """
        Perform walk-forward testing with expanding training window and fixed test window
        
        Args:
            df: DataFrame with price data
            min_train_years: Minimum number of years for initial training (default: 3)
            max_train_years: Maximum number of years for training (default: 7)
            test_years: Number of years to use for testing (default: 1)
            output_dir: Directory to save results
        """
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get current timestamp for this walk-forward test
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be datetime")
        
        # Calculate total number of days in the dataset
        total_days = (df.index[-1] - df.index[0]).days
        print(f"Total days in dataset: {total_days}")
        
        # Initialize results storage
        walk_forward_results = []
        
        # Perform walk-forward testing for different training periods
        for train_years in range(min_train_years, max_train_years + 1):
            # Calculate training period end date
            train_end_date = df.index[0] + timedelta(days=train_years*365)
            
            # Calculate test period end date (1 year after training end)
            test_end_date = train_end_date + timedelta(days=test_years*365)
            
            # Split data into training and testing sets
            train_data = df[df.index <= train_end_date]
            test_data = df[(df.index > train_end_date) & (df.index <= test_end_date)]
            
            if len(test_data) == 0:
                print(f"Skipping {train_years} years training period - no test data available")
                continue
                
            print(f"\nTraining period: {train_years} years")
            print(f"Training data: {train_data.index[0]} to {train_data.index[-1]}")
            print(f"Testing data: {test_data.index[0]} to {test_data.index[-1]}")
            print(f"Training days: {len(train_data)}, Testing days: {len(test_data)}")
            
            # Create period directory
            period_dir = os.path.join(output_dir, f'train_{train_years}_years_{timestamp}')
            os.makedirs(period_dir, exist_ok=True)
            
            # Create train and test subdirectories
            train_dir = os.path.join(period_dir, 'training_results')
            test_dir = os.path.join(period_dir, 'testing_results')
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)
            
            # Create a new model instance for this training period
            period_model = MarketRegimeHMM(
                n_components=4,
                covariance_type='diag',
                n_iter=self.model.n_iter,
                scaling_method=self.scaling_method,
                init_method='random'
            )
            
            # Train model on training data
            period_model.fit(train_data)
            
            # Get predictions for both training and test data
            train_regimes, train_states, train_features = period_model.predict(train_data)
            test_regimes, test_states, test_features = period_model.predict(test_data)
            
            # Calculate statistics for both periods
            train_stats = period_model.get_regime_statistics(train_data)
            test_stats = period_model.get_regime_statistics(test_data)
            
            # Calculate transition matrix
            transition_matrix = period_model.get_transition_matrix()
            
            # Store results
            period_results = {
                'train_years': train_years,
                'train_start': train_data.index[0],
                'train_end': train_data.index[-1],
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1],
                'transition_matrix': transition_matrix,
                'train_sequential_stats': train_stats['sequential'],
                'train_aggregated_stats': train_stats['aggregated'],
                'test_sequential_stats': test_stats['sequential'],
                'test_aggregated_stats': test_stats['aggregated'],
                'train_regime_distribution': pd.Series(train_regimes).value_counts(),
                'test_regime_distribution': pd.Series(test_regimes).value_counts(),
                'model_parameters': {
                    'n_components': period_model.n_components,
                    'covariance_type': period_model.model.covariance_type,
                    'n_iter': period_model.model.n_iter,
                    'scaling_method': period_model.scaling_method
                }
            }
            
            walk_forward_results.append(period_results)
            
            # Save training results
            transition_matrix.to_csv(os.path.join(train_dir, 'transition_matrix.csv'))
            train_stats['sequential'].to_csv(os.path.join(train_dir, 'sequential_statistics.csv'))
            train_stats['aggregated'].to_csv(os.path.join(train_dir, 'aggregated_statistics.csv'))
            pd.Series(train_regimes).value_counts().to_csv(os.path.join(train_dir, 'regime_distribution.csv'))
            train_fig = period_model.plot_regimes(train_data)
            train_fig.write_html(os.path.join(train_dir, 'regime_plot.html'))
            
            # Save testing results
            test_stats['sequential'].to_csv(os.path.join(test_dir, 'sequential_statistics.csv'))
            test_stats['aggregated'].to_csv(os.path.join(test_dir, 'aggregated_statistics.csv'))
            pd.Series(test_regimes).value_counts().to_csv(os.path.join(test_dir, 'regime_distribution.csv'))
            test_fig = period_model.plot_regimes(test_data)
            test_fig.write_html(os.path.join(test_dir, 'regime_plot.html'))
            
            # Save model parameters for this period
            period_model.save_model(os.path.join(period_dir, 'model'))
            
            print(f"Results saved to {period_dir}")
            print(f"- Training results: {train_dir}")
            print(f"- Testing results: {test_dir}")
        
        # Create summary of all periods
        summary = pd.DataFrame([{
            'train_years': r['train_years'],
            'train_start': r['train_start'],
            'train_end': r['train_end'],
            'test_start': r['test_start'],
            'test_end': r['test_end'],
            'test_days': (r['test_end'] - r['test_start']).days,
            'train_regime_counts': len(r['train_regime_distribution']),
            'test_regime_counts': len(r['test_regime_distribution']),
            'train_most_common_regime': r['train_regime_distribution'].index[0],
            'train_most_common_regime_count': r['train_regime_distribution'].iloc[0],
            'test_most_common_regime': r['test_regime_distribution'].index[0],
            'test_most_common_regime_count': r['test_regime_distribution'].iloc[0],
            'covariance_type': r['model_parameters']['covariance_type'],
            'n_iter': r['model_parameters']['n_iter'],
            'scaling_method': r['model_parameters']['scaling_method']
        } for r in walk_forward_results])
        
        # Save summary
        summary.to_csv(os.path.join(output_dir, f'walk_forward_summary_{timestamp}.csv'))
        
        return walk_forward_results

    def save_model(self, path):
        """
        Save the model and scalers to disk
        
        Args:
            path: Directory path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
            
        os.makedirs(path, exist_ok=True)
        
        # Save the HMM model
        joblib.dump(self.model, os.path.join(path, 'hmm_model.joblib'))
        
        # Save the scalers
        joblib.dump(self.scalers, os.path.join(path, 'scalers.joblib'))
        
        # Save other attributes
        model_info = {
            'n_components': self.n_components,
            'scaling_method': self.scaling_method,
            'feature_names': self.feature_names,
            'state_to_regime': self.state_to_regime,
            'is_fitted': self.is_fitted
        }
        joblib.dump(model_info, os.path.join(path, 'model_info.joblib'))
    
    @classmethod
    def load_model(cls, path):
        """
        Load a saved model from disk
        
        Args:
            path: Directory path containing the saved model
        """
        # Load the HMM model
        model = joblib.load(os.path.join(path, 'hmm_model.joblib'))
        
        # Load the scalers
        scalers = joblib.load(os.path.join(path, 'scalers.joblib'))
        
        # Load model info
        model_info = joblib.load(os.path.join(path, 'model_info.joblib'))
        
        # Create new instance
        instance = cls(
            n_components=model_info['n_components'],
            scaling_method=model_info['scaling_method']
        )
        
        # Restore attributes
        instance.model = model
        instance.scalers = scalers
        instance.feature_names = model_info['feature_names']
        instance.state_to_regime = model_info['state_to_regime']
        instance.is_fitted = model_info['is_fitted']
        
        return instance

    def get_regularization_info(self):
        """
        Get information about current regularization settings
        """
        return {
            'min_covar': self.min_covar,
            'covariance_type': self.covariance_type,
            'scaling_method': self.scaling_method,
            'n_components': self.n_components
        }
        
    def set_regularization(self, min_covar=None, covariance_type=None, scaling_method=None):
        """
        Update regularization parameters
        
        Args:
            min_covar: New minimum covariance value
            covariance_type: New covariance type
            scaling_method: New scaling method
        """
        if min_covar is not None:
            self.min_covar = min_covar
            self.model.min_covar = min_covar
            
        if covariance_type is not None:
            self.covariance_type = covariance_type
            self.model.covariance_type = covariance_type
            
        if scaling_method is not None:
            self.scaling_method = scaling_method
            
        # Reset fitted state as parameters have changed
        self.is_fitted = False

# Example usage and testing
def load_sample_data():
    """
    Load or generate sample market data for testing
    """
    dates = pd.to_datetime(BTC_data['date'])
    
    # Generate OHLCV data
    df = pd.DataFrame(index=dates)
    df['Close'] = BTC_data['close']
    df['Open'] = BTC_data['open']
    df['High'] = BTC_data['high']
    df['Low'] = BTC_data['low']
    df['Volume'] = BTC_data['volume']
    
    print(df.head())
    return df.dropna()

def main():
    """
    Main function to demonstrate HMM regime detection
    """
    # Load data
    df = BTC_data.copy()
    
    # Set datetime index
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Rename columns to match expected format
    df['Close'] = df['close']
    df['Open'] = df['open']
    df['High'] = df['high']
    df['Low'] = df['low']
    df['Volume'] = df['volume']
    
    print("Data loaded successfully!")
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Initialize HMM model
    hmm_model = MarketRegimeHMM(n_components=4)
    
    # Train on entire dataset (no testing)
    print("Training on full dataset...")
    hmm_model.fit(df)
    
    # Save training-only results and model
    results = hmm_model.save_results(df, output_dir='hmm_training_results')
    hmm_model.save_model(os.path.join('hmm_training_results', 'model'))
    
    return hmm_model, df, results

if __name__ == "__main__":
    model, data, results = main()
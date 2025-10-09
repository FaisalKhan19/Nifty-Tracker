import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass
from datetime import datetime

@dataclass
class OrderBlock:
    """Class to represent an order block"""
    left_time: datetime
    top: float
    bottom: float
    avg: float
    ob_type: str  # 'bullish' or 'bearish'
    is_mitigated: bool = False
    mitigation_time: Optional[datetime] = None

class OrderBlockDetector:
    def __init__(self, 
                 volume_pivot_length: int = 5,
                 bullish_ob_count: int = 3,
                 bearish_ob_count: int = 3,
                 mitigation_method: str = 'wick'):
        """
        Initialize Order Block Detector
        
        Parameters:
        - volume_pivot_length: Length for volume pivot detection
        - bullish_ob_count: Number of bullish order blocks to track
        - bearish_ob_count: Number of bearish order blocks to track
        - mitigation_method: 'wick' or 'close' for mitigation detection
        """
        self.length = volume_pivot_length
        self.bull_ext_last = bullish_ob_count
        self.bear_ext_last = bearish_ob_count
        self.mitigation_method = mitigation_method
        
        # Storage for order blocks
        self.bullish_obs: List[OrderBlock] = []
        self.bearish_obs: List[OrderBlock] = []
        
    def _get_pivot_high(self, series: pd.Series, length: int) -> pd.Series:
        """Detect pivot highs in a series"""
        pivots = pd.Series(index=series.index, dtype=float)
        
        for i in range(length, len(series) - length):
            current = series.iloc[i]
            left_max = series.iloc[i-length:i].max()
            right_max = series.iloc[i+1:i+length+1].max()
            
            if current > left_max and current > right_max:
                pivots.iloc[i] = current
                
        return pivots
    
    def _get_rolling_extremes(self, df: pd.DataFrame, length: int) -> Tuple[pd.Series, pd.Series]:
        """Get rolling highest and lowest values"""
        upper = df['high'].rolling(window=length*2+1, center=True).max()
        lower = df['low'].rolling(window=length*2+1, center=True).min()
        return upper, lower
    
    def _determine_market_structure(self, df: pd.DataFrame, length: int) -> pd.Series:
        """Determine market structure (0 = bullish, 1 = bearish)"""
        upper, lower = self._get_rolling_extremes(df, length)
        os = pd.Series(index=df.index, dtype=int)
        
        current_os = 0
        for i in range(length, len(df)):
            if i >= length and df['high'].iloc[i-length] > upper.iloc[i-length]:
                current_os = 0  # Bullish structure
            elif i >= length and df['low'].iloc[i-length] < lower.iloc[i-length]:
                current_os = 1  # Bearish structure
            
            os.iloc[i] = current_os
            
        return os
    
    def _get_mitigation_targets(self, df: pd.DataFrame, length: int) -> Tuple[pd.Series, pd.Series]:
        """Get mitigation target levels"""
        if self.mitigation_method == 'close':
            target_bull = df['close'].rolling(window=length*2+1, center=True).min()
            target_bear = df['close'].rolling(window=length*2+1, center=True).max()
        else:  # wick
            target_bull = df['low'].rolling(window=length*2+1, center=True).min()
            target_bear = df['high'].rolling(window=length*2+1, center=True).max()
            
        return target_bull, target_bear
    
    def _check_mitigation(self, ob: OrderBlock, current_price: float, current_time: datetime) -> bool:
        """Check if an order block is mitigated"""
        if ob.is_mitigated:
            return True
            
        if ob.ob_type == 'bullish':
            # Bullish OB mitigated when price goes below bottom
            if current_price < ob.bottom:
                ob.is_mitigated = True
                ob.mitigation_time = current_time
                return True
        else:  # bearish
            # Bearish OB mitigated when price goes above top
            if current_price > ob.top:
                ob.is_mitigated = True
                ob.mitigation_time = current_time
                return True
                
        return False
    
    def detect_order_blocks(self, df: pd.DataFrame) -> Dict:
        """
        Main function to detect order blocks
        
        Parameters:
        - df: DataFrame with columns ['date', 'open', 'high', 'low', 'close', 'volume']
        
        Returns:
        - Dictionary with detected order blocks and signals
        """
        # Ensure date is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        elif df.index.name != 'date':
            df.index = pd.to_datetime(df.index)
            
        # Calculate hl2 (average of high and low)
        df['hl2'] = (df['high'] + df['low']) / 2
        
        # Get volume pivot highs
        volume_pivots = self._get_pivot_high(df['volume'], self.length)
        
        # Determine market structure
        market_structure = self._determine_market_structure(df, self.length)
        
        # Get mitigation targets
        target_bull, target_bear = self._get_mitigation_targets(df, self.length)
        
        # Storage for signals
        bull_signals = []
        bear_signals = []
        mitigation_signals = []
        
        # Process each bar
        for i in range(self.length, len(df)):
            current_time = df.index[i]
            current_bar_index = i - self.length
            
            # Check for volume pivot at the lagged position
            if not pd.isna(volume_pivots.iloc[current_bar_index]):
                pivot_time = df.index[current_bar_index]
                
                # Bullish Order Block (volume pivot in bearish structure)
                if market_structure.iloc[current_bar_index] == 1:
                    top = df['hl2'].iloc[current_bar_index]
                    bottom = df['low'].iloc[current_bar_index]
                    avg = (top + bottom) / 2
                    
                    ob = OrderBlock(
                        left_time=pivot_time,
                        top=top,
                        bottom=bottom,
                        avg=avg,
                        ob_type='bullish'
                    )
                    
                    # Remove old OBs if we exceed the limit
                    if len(self.bullish_obs) >= self.bull_ext_last:
                        self.bullish_obs.pop(0)
                    
                    self.bullish_obs.append(ob)
                    bull_signals.append({
                        'date': current_time,
                        'price': bottom,
                        'type': 'bullish_ob_formed'
                    })
                
                # Bearish Order Block (volume pivot in bullish structure)
                elif market_structure.iloc[current_bar_index] == 0:
                    top = df['high'].iloc[current_bar_index]
                    bottom = df['hl2'].iloc[current_bar_index]
                    avg = (top + bottom) / 2
                    
                    ob = OrderBlock(
                        left_time=pivot_time,
                        top=top,
                        bottom=bottom,
                        avg=avg,
                        ob_type='bearish'
                    )
                    
                    # Remove old OBs if we exceed the limit
                    if len(self.bearish_obs) >= self.bear_ext_last:
                        self.bearish_obs.pop(0)
                    
                    self.bearish_obs.append(ob)
                    bear_signals.append({
                        'date': current_time,
                        'price': top,
                        'type': 'bearish_ob_formed'
                    })
            
            # Check for mitigation of existing order blocks
            mitigation_price = target_bull.iloc[i] if self.mitigation_method == 'close' else df['low'].iloc[i]
            
            # Check bullish OB mitigation
            for ob in self.bullish_obs:
                if self._check_mitigation(ob, mitigation_price, current_time):
                    mitigation_signals.append({
                        'date': current_time,
                        'price': mitigation_price,
                        'type': 'bullish_ob_mitigated',
                        'ob_time': ob.left_time
                    })
            
            # Check bearish OB mitigation
            mitigation_price = target_bear.iloc[i] if self.mitigation_method == 'close' else df['high'].iloc[i]
            for ob in self.bearish_obs:
                if self._check_mitigation(ob, mitigation_price, current_time):
                    mitigation_signals.append({
                        'date': current_time,
                        'price': mitigation_price,
                        'type': 'bearish_ob_mitigated',
                        'ob_time': ob.left_time
                    })
        
        return {
            'bullish_order_blocks': [ob for ob in self.bullish_obs if not ob.is_mitigated],
            'bearish_order_blocks': [ob for ob in self.bearish_obs if not ob.is_mitigated],
            'bull_signals': bull_signals,
            'bear_signals': bear_signals,
            'mitigation_signals': mitigation_signals,
            'all_bullish_obs': self.bullish_obs,
            'all_bearish_obs': self.bearish_obs
        }
    
    def plot_order_blocks(self, df: pd.DataFrame, results: Dict, 
                         start_date: Optional[str] = None, 
                         end_date: Optional[str] = None):
        """
        Plot candlestick chart with order blocks
        """
        # Filter data by date range if provided
        plot_df = df.copy()
        if start_date:
            plot_df = plot_df[plot_df.index >= start_date]
        if end_date:
            plot_df = plot_df[plot_df.index <= end_date]
        
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Plot candlesticks
        for i, (date, row) in enumerate(plot_df.iterrows()):
            color = 'green' if row['close'] >= row['open'] else 'red'
            
            # Candle body
            height = abs(row['close'] - row['open'])
            bottom = min(row['open'], row['close'])
            
            rect = patches.Rectangle((i, bottom), 0.8, height, 
                                   linewidth=1, edgecolor=color, 
                                   facecolor=color, alpha=0.7)
            ax.add_patch(rect)
            
            # Wicks
            ax.plot([i+0.4, i+0.4], [row['low'], row['high']], color=color, linewidth=1)
        
        # Plot active bullish order blocks
        for ob in results['bullish_order_blocks']:
            if ob.left_time in plot_df.index:
                start_idx = plot_df.index.get_loc(ob.left_time)
                end_idx = len(plot_df) - 1
                
                rect = patches.Rectangle((start_idx, ob.bottom), 
                                       end_idx - start_idx, ob.top - ob.bottom,
                                       linewidth=1, edgecolor='green', 
                                       facecolor='green', alpha=0.2)
                ax.add_patch(rect)
                
                # Average line
                ax.plot([start_idx, end_idx], [ob.avg, ob.avg], 
                       color='darkgreen', linestyle='--', alpha=0.7)
        
        # Plot active bearish order blocks
        for ob in results['bearish_order_blocks']:
            if ob.left_time in plot_df.index:
                start_idx = plot_df.index.get_loc(ob.left_time)
                end_idx = len(plot_df) - 1
                
                rect = patches.Rectangle((start_idx, ob.bottom), 
                                       end_idx - start_idx, ob.top - ob.bottom,
                                       linewidth=1, edgecolor='red', 
                                       facecolor='red', alpha=0.2)
                ax.add_patch(rect)
                
                # Average line
                ax.plot([start_idx, end_idx], [ob.avg, ob.avg], 
                       color='darkred', linestyle='--', alpha=0.7)
        
        # Plot formation signals
        for signal in results['bull_signals']:
            if signal['date'] in plot_df.index:
                idx = plot_df.index.get_loc(signal['date'])
                ax.scatter(idx, signal['price'], color='green', marker='^', s=100, alpha=0.8)
        
        for signal in results['bear_signals']:
            if signal['date'] in plot_df.index:
                idx = plot_df.index.get_loc(signal['date'])
                ax.scatter(idx, signal['price'], color='red', marker='v', s=100, alpha=0.8)
        
        ax.set_title('Order Block Detection')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.grid(True, alpha=0.3)
        
        # Set x-axis labels
        x_ticks = np.arange(0, len(plot_df), max(1, len(plot_df)//10))
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([plot_df.index[i].strftime('%Y-%m-%d') for i in x_ticks], rotation=45)
        
        plt.tight_layout()
        plt.show()

def load_and_detect_order_blocks(csv_path: str, 
                                volume_pivot_length: int = 5,
                                bullish_ob_count: int = 3,
                                bearish_ob_count: int = 3,
                                mitigation_method: str = 'wick'):
    """
    Load CSV data and detect order blocks
    
    Parameters:
    - csv_path: Path to CSV file with OHLCV data
    - volume_pivot_length: Length for volume pivot detection
    - bullish_ob_count: Number of bullish order blocks to track
    - bearish_ob_count: Number of bearish order blocks to track
    - mitigation_method: 'wick' or 'close'
    
    Returns:
    - Dictionary with results and DataFrame
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    # Ensure proper column names (adjust as needed for your CSV format)
    expected_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in expected_columns):
        print(f"Expected columns: {expected_columns}")
        print(f"Found columns: {list(df.columns)}")
        raise ValueError("CSV must contain columns: date, open, high, low, close, volume")
    
    # Initialize detector
    detector = OrderBlockDetector(
        volume_pivot_length=volume_pivot_length,
        bullish_ob_count=bullish_ob_count,
        bearish_ob_count=bearish_ob_count,
        mitigation_method=mitigation_method
    )
    
    # Detect order blocks
    results = detector.detect_order_blocks(df)
    
    print(f"Detected {len(results['bullish_order_blocks'])} active bullish order blocks")
    print(f"Detected {len(results['bearish_order_blocks'])} active bearish order blocks")
    print(f"Total bullish signals: {len(results['bull_signals'])}")
    print(f"Total bearish signals: {len(results['bear_signals'])}")
    print(f"Total mitigation signals: {len(results['mitigation_signals'])}")
    
    return results, df, detector

# Example usage
if __name__ == "__main__":
    # Example usage:
    results, df, detector = load_and_detect_order_blocks('ohlc_weekly_data/AMBUJACEM_weekly.csv')
    # detector.plot_order_blocks(df, results)
    import json
    with open("lux_algo_ob.json", 'w') as f:
        json.dump(results, f)
    print("Order Block Detector ready!")
    print("Usage example:")
    print("results, df, detector = load_and_detect_order_blocks('your_data.csv')")
    print("detector.plot_order_blocks(df, results)")

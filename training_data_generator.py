import pandas as pd
import pandas_ta as ta
import numpy as np  
import mplfinance as mpf
import os
import matplotlib.pyplot as plt

# Load your data
data = pd.read_csv('gc_trading_1h.csv', index_col=0, parse_dates=True)

# Calculate the EMAs
data['EMA6_close'] = ta.ema(data['close'], length=6)
data['EMA21_close'] = ta.ema(data['close'], length=21)
data['EMA6_high'] = ta.ema(data['high'], length=6)
data['EMA6_low'] = ta.ema(data['low'], length=6)

# Function to classify candles based on candle i+2
def classify_candle(df, i):
    if df['close'].iloc[i + 1] > df['close'].iloc[i]:
        return 'bull'
    elif df['close'].iloc[i + 1] < df['close'].iloc[i]:
        return 'bear'
    else:
        return 'range'

# Create folders for bull and bear if not exist
if not os.path.exists('bull'):
    os.makedirs('bull')
if not os.path.exists('bear'):
    os.makedirs('bear')
if not os.path.exists('range'):
    os.makedirs('range')

# Generate and save charts
for i in range(21, len(data) - 2):  # Prevent going out of index

    # Select two candles
    subset = data.iloc[i:i+1]

    # Skip if the subset is empty or invalid
    if subset.empty or len(subset) < 1:
        continue

    # Classify the next candle
    label = classify_candle(data, i)

    # Plot the candlestick chart
    mc = mpf.make_marketcolors(up='#166138', down='#b11e31', wick='black', edge='black')
    s  = mpf.make_mpf_style(marketcolors=mc)

    # Plot EMAs as dots (markers)
    add_plot = [
        mpf.make_addplot(subset['EMA6_close'], color='#ff7350', scatter=True, markersize=300),  # Orange
        mpf.make_addplot(subset['EMA21_close'], color='#42cde7', scatter=True, markersize=300),  # Deep Ocean Blue
        mpf.make_addplot(subset['EMA6_high'], color='#3e3c3c', scatter=True, markersize=300),   # cream
        mpf.make_addplot(subset['EMA6_low'], color='#ffd159', scatter=True, markersize=300)     # Yellow
    ]
 # Set figure size (taller and narrower) and hide the axes
    fig, ax = mpf.plot(subset, type='candle', style=s, addplot=add_plot, figsize=(4, 8), returnfig=True)
    ax[0].axis('off')  # Turn off the axes

    # Save plot in the corresponding folder
    save_path = f'{label}/chart_{i}.png'
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)

    print(f'Saved chart {i} as {label}')
    plt.close(fig)  # Close the figure to free up memory
#Joan Chepkwony's Cryptocurrecncy Agent
import requests
import pandas as pd
import time
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()

# Initialize Groq client
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# List of cryptocurrencies we want to trade
CRYPTOCURRENCIES = ['bitcoin', 'ethereum']

# URL for historical market data from CoinGecko (free API)
API_URL = 'https://api.coingecko.com/api/v3/coins/{coin}/market_chart'

# Function to get historical data (last 90 days)
def fetch_price_history(coin='bitcoin', days=90):
    params = {
        'vs_currency': 'usd',
        'days': days,
        'interval': 'daily'
    }
    response = requests.get(API_URL.format(coin=coin), params=params)
    data = response.json()
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

# Simulate simple moving average crossover strategy
def simulate_trading(df, short_window=20, long_window=50, initial_cash=1000):
    df['short_ma'] = df['price'].rolling(window=short_window).mean()
    df['long_ma'] = df['price'].rolling(window=long_window).mean()

    df.dropna(inplace=True)
    position = 0
    cash = initial_cash
    trades = []

    for i in range(1, len(df)):
        prev_short = df['short_ma'].iloc[i - 1]
        prev_long = df['long_ma'].iloc[i - 1]
        curr_short = df['short_ma'].iloc[i]
        curr_long = df['long_ma'].iloc[i]
        price = df['price'].iloc[i]

        if prev_short < prev_long and curr_short > curr_long:
            # Buy
            if cash > 0:
                position = cash / price
                cash = 0
                trades.append((df.index[i], 'BUY', price))

        elif prev_short > prev_long and curr_short < curr_long:
            # Sell
            if position > 0:
                cash = position * price
                position = 0
                trades.append((df.index[i], 'SELL', price))

    # Final value
    final_value = cash + position * df['price'].iloc[-1]
    return final_value, trades, df

# Visualize trades
def plot_trades(df, trades, title='Trading Chart'):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['price'], label='Price', alpha=0.6)
    for t in trades:
        color = 'green' if t[1] == 'BUY' else 'red'
        plt.axvline(t[0], color=color, linestyle='--', alpha=0.7, label=t[1])
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# Run the bot for both BTC and ETH
if __name__ == '__main__':
    for coin in CRYPTOCURRENCIES:
        print(f"\nRunning bot for {coin.title()}...")
        df = fetch_price_history(coin)
        final_value, trades, df = simulate_trading(df)
        print(f"Initial cash: $1000 | Final value: ${final_value:.2f}")
        print("Trades:")
        for t in trades:
            print(f"  {t[0].strftime('%Y-%m-%d')} - {t[1]} at ${t[2]:.2f}")
        plot_trades(df, trades, title=f"{coin.title()} Strategy")

        # Example of using Groq for additional insights
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"Provide insights on trading strategies for {coin}.",
                }
            ],
            model="llama-3.3-70b-versatile",
        )
        print("Groq Insights:", chat_completion.choices[0].message.content)
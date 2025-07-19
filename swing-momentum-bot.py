import yfinance as yf import pandas as pd import streamlit as st import datetime import plotly.graph_objects as go from plotly.subplots import make_subplots import logging

Configure logging

logging.basicConfig(level=logging.INFO) logger = logging.getLogger(name)

st.set_page_config(page_title="Swing Momentum Bot", layout="wide")

Configuration Class

class TradingConfig: BUY_THRESHOLD = 50 SELL_THRESHOLD = 70 MIN_VOLUME = 500000 MACD_FAST = 12 MACD_SLOW = 26 MACD_SIGNAL = 9 BB_PERIOD = 20 BB_STD = 2 DELTA_TARGET = 0.4 ENTRY_TIME = "15:45" EXIT_TIME = "09:31"

config = TradingConfig()

TICKERS = ['AAPL', 'MSFT', 'NVDA', 'AMD', 'TSLA', 'GOOGL', 'AMZN', 'META', 'JPM', 'V']

st.title("\ud83d\udcc8 Swing Momentum Bot Dashboard") st.markdown("Scans ~100 tickers for overnight swing trades with momentum indicators and call option suggestions.")

@st.cache_data(ttl=60) def fetch_data(ticker): try: df = yf.download(ticker, period='1d', interval='1m', progress=False) if df.empty or len(df) < 50: return None df = calculate_technical_indicators(df) return df except Exception as e: logger.error(f"Error fetching data for {ticker}: {e}") return None

def calculate_technical_indicators(df): try: df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean() delta = df['Close'].diff() gain = delta.where(delta > 0, 0.0) loss = -delta.where(delta < 0, 0.0) avg_gain = gain.rolling(14).mean() avg_loss = loss.rolling(14).mean() rs = avg_gain / avg_loss df['RSI'] = 100 - (100 / (1 + rs.replace(float('inf'), 0))) ema_fast = df['Close'].ewm(span=config.MACD_FAST, adjust=False).mean() ema_slow = df['Close'].ewm(span=config.MACD_SLOW, adjust=False).mean() df['MACD'] = ema_fast - ema_slow df['MACD_Signal'] = df['MACD'].ewm(span=config.MACD_SIGNAL, adjust=False).mean() df['MACD_Hist'] = df['MACD'] - df['MACD_Signal'] df['BB_Mid'] = df['Close'].rolling(config.BB_PERIOD).mean() df['BB_Std'] = df['Close'].rolling(config.BB_PERIOD).std() df['BB_Upper'] = df['BB_Mid'] + config.BB_STD * df['BB_Std'] df['BB_Lower'] = df['BB_Mid'] - config.BB_STD * df['BB_Std'] return df except Exception as e: logger.error(f"Error calculating indicators: {e}") return None

@st.cache_data(ttl=300) def fetch_options(ticker): try: stock = yf.Ticker(ticker) expirations = stock.options if not expirations: return None expiration = expirations[0] calls = stock.option_chain(expiration).calls return calls except Exception as e: logger.error(f"Error fetching options for {ticker}: {e}") return None

def find_best_call_option(options, current_price): if options is None or options.empty: return None try: reasonable = options[(options['strike'] >= current_price * 0.9) & (options['strike'] <= current_price * 1.1)] if reasonable.empty: return None reasonable = reasonable.copy() reasonable['strike_diff'] = abs(reasonable['strike'] - current_price) best_option = reasonable.loc[reasonable['strike_diff'].idxmin()] return round(best_option['strike'], 2) except Exception as e: logger.error(f"Error selecting option: {e}") return None

def evaluate_trade(df, ticker): if df is None or len(df) < 50: return None try: latest = df.iloc[-1] if latest['Volume'] < config.MIN_VOLUME: return None required = ['RSI', 'Close', 'EMA20', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower'] if latest[required].isna().any(): return None rsi, price, ema20 = latest['RSI'], latest['Close'], latest['EMA20'] macd, macd_sig = latest['MACD'], latest['MACD_Signal'] bb_u, bb_l, vol = latest['BB_Upper'], latest['BB_Lower'], latest['Volume'] signal, reason, strike, score = "WAIT", "", None, 0 if (rsi < config.BUY_THRESHOLD and price > ema20 and macd > macd_sig and price < bb_l + (bb_u - bb_l) * 0.2): signal = "BUY" reason = f"RSI {rsi:.2f} < {config.BUY_THRESHOLD}, Price > EMA20, MACD Bullish, Near BB Lower" score = config.BUY_THRESHOLD - rsi opts = fetch_options(ticker) strike = find_best_call_option(opts, price) elif (rsi > config.SELL_THRESHOLD and price < ema20 and macd < macd_sig): signal = "SELL" reason = f"RSI {rsi:.2f} > {config.SELL_THRESHOLD}, Price < EMA20, MACD Bearish" if signal == "WAIT": return None return { 'Symbol': ticker, 'Signal': signal, 'RSI': round(rsi, 2), 'Close': round(price, 2), 'EMA20': round(ema20, 2), 'MACD': round(macd, 2), 'MACD_Signal': round(macd_sig, 2), 'BB_Upper': round(bb_u, 2), 'BB_Lower': round(bb_l, 2), 'Volume': int(vol), 'Call_Strike': strike, 'Reason': reason, 'Signal_Score': score } except Exception as e: logger.error(f"Error in trade evaluation for {ticker}: {e}") return None

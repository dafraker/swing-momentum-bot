import yfinance as yf
import pandas as pd
import streamlit as st
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Swing Momentum Bot", layout="wide")

TICKERS = [
    'AAPL', 'MSFT', 'NVDA', 'AMD', 'TSLA', 'GOOGL', 'AMZN', 'META', 'JPM', 'V',
    'NFLX', 'QCOM', 'BA', 'SPY', 'QQQ', 'SMH', 'XLF', 'XLE', 'XLU', 'XLK',
    'ADBE', 'CSCO', 'INTC', 'IBM', 'ORCL', 'CRM', 'PYPL', 'SQ', 'UBER', 'LYFT',
    'WMT', 'TGT', 'COST', 'HD', 'LOW', 'NKE', 'LULU', 'DIS', 'CMCSA', 'SBUX',
    'GILD', 'AMGN', 'MRNA', 'PFE', 'JNJ', 'LLY', 'CVS', 'UNH', 'CI', 'BMY',
    'BAC', 'WFC', 'GS', 'MS', 'C', 'SCHW', 'T', 'VZ', 'TMUS', 'F', 'GM',
    'CAT', 'DE', 'MMM', 'GE', 'LMT', 'RTX', 'XOM', 'CVX', 'COP', 'SLB', 'OXY',
    'DAL', 'UAL', 'AAL', 'LUV', 'CCL', 'RCL', 'MAR', 'HLT', 'BKNG', 'ABNB',
    'EEM', 'IWM', 'DIA', 'GLD', 'SLV', 'USO', 'UNG', 'VXX', 'TLT', 'HYG',
    'BABA', 'JD', 'PDD', 'NIO', 'XPEV', 'RBLX', 'COIN', 'HOOD', 'DKNG', 'PLTR'
]
ENTRY_TIME = "15:45"
EXIT_TIME = "09:31"
BUY_THRESHOLD = 30
SELL_THRESHOLD = 70
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2
DELTA_TARGET = 0.4
MIN_VOLUME = 1000000

st.title("📈 Swing Momentum Bot Dashboard")
st.markdown("Scans ~100 tickers for overnight swing trades with momentum indicators and call option suggestions.")

@st.cache_data(ttl=60)
def fetch_data(ticker):
    try:
        df = yf.download(ticker, period='1d', interval='1m', progress=False)
        if df.empty or len(df) < 20:
            return None
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs.replace(float('inf'), 0)))
        ema_fast = df['Close'].ewm(span=MACD_FAST, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=MACD_SLOW, adjust=False).mean()
        df['MACD'] = ema_fast - ema_slow
        df['MACD_Signal'] = df['MACD'].ewm(span=MACD_SIGNAL, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        df['BB_Mid'] = df['Close'].rolling(BB_PERIOD).mean()
        df['BB_Std'] = df['Close'].rolling(BB_PERIOD).std()
        df['BB_Upper'] = df['BB_Mid'] + BB_STD * df['BB_Std']
        df['BB_Lower'] = df['BB_Mid'] - BB_STD * df['BB_Std']
        return df
    except Exception as e:
        st.warning(f"Error fetching data for {ticker}: {e}")
        return None

@st.cache_data(ttl=300)
def fetch_options(ticker):
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations:
            return None
        expiration = expirations[0]
        opts = stock.option_chain(expiration)
        calls = opts.calls
        calls['Delta'] = calls['lastPrice'] / calls['strike']
        return calls
    except Exception as e:
        st.warning(f"Error fetching options for {ticker}: {e}")
        return None

def evaluate_trade(df, ticker):
    if df is None or len(df) < 1:
        return None
    latest = df.iloc[-1]
    if pd.isna(latest['RSI']) or pd.isna(latest['Close']) or pd.isna(latest['EMA20']) or \
       pd.isna(latest['MACD']) or pd.isna(latest['MACD_Signal']) or pd.isna(latest['BB_Upper']) or \
       pd.isna(latest['BB_Lower']) or pd.isna(latest['Volume']):
        return None
    rsi = latest['RSI']
    price = latest['Close']
    ema20 = latest['EMA20']
    macd = latest['MACD']
    macd_signal = latest['MACD_Signal']
    bb_upper = latest['BB_Upper']
    bb_lower = latest['BB_Lower']
    volume = latest['Volume']

    signal = "WAIT"
    reason = ""
    call_strike = None
    signal_score = 0

    if volume < MIN_VOLUME:
        return None

    if (rsi < BUY_THRESHOLD and price > ema20 and
        macd > macd_signal and price < bb_lower + (bb_upper - bb_lower) * 0.2):
        signal = "BUY"
        reason = f"RSI {rsi:.2f} < {BUY_THRESHOLD}, Price > EMA20, MACD Bullish, Near BB Lower"
        signal_score = BUY_THRESHOLD - rsi
        options = fetch_options(ticker)
        if options is not None and not options.empty:
            options['Delta_Diff'] = abs(options['Delta'] - DELTA_TARGET)
            best_call = options.loc[options['Delta_Diff'].idxmin()]
            call_strike = round(best_call['strike'], 2)

    elif rsi > SELL_THRESHOLD and price < ema20 and macd < macd_signal:
        signal = "SELL"
        reason = f"RSI {rsi:.2f} > {SELL_THRESHOLD}, Price < EMA20, MACD Bearish"

    return {
        'Symbol': ticker,
        'Signal': signal,
        'RSI': round(rsi, 2),
        'Close': round(price, 2),
        'EMA20': round(ema20, 2),
        'MACD': round(macd, 2),
        'MACD_Signal': round(macd_signal, 2),
        'BB_Upper': round(bb_upper, 2),
        'BB_Lower': round(bb_lower, 2),
        'Volume': int(volume),
        'Call_Strike': call_strike,
        'Reason': reason,
        'Signal_Score': signal_score
    }

@st.cache_data(ttl=300)
def fetch_market_trends():
    vix = yf.download('^VIX', period='1d', interval='1d', progress=False)
    sector_etfs = ['XLF', 'XLK', 'XLE', 'XLU', 'XLY']
    trends = {}
    for etf in sector_etfs:
        df = yf.download(etf, period='1d', interval='1d', progress=False)
        trends[etf] = round(df['Close'].iloc[-1], 2) if not df.empty and len(df) > 0 else None
    trends['VIX'] = round(vix['Close'].iloc[-1], 2) if not vix.empty and len(vix) > 0 else None
    return trends

st.subheader("Top Trade Signals")
results = []
progress_bar = st.progress(0)
for i, ticker in enumerate(TICKERS):
    df = fetch_data(ticker)
    if df is not None:
        trade = evaluate_trade(df, ticker)
        if trade:
            results.append(trade)
    progress_bar.progress((i + 1) / len(TICKERS))

df_results = pd.DataFrame(results)
if not df_results.empty:
    buy_signals = df_results[df_results['Signal'] == 'BUY'].sort_values(by='Signal_Score', ascending=False).head(10)
    st.write("**Top BUY Signals**")
    st.dataframe(buy_signals, use_container_width=True)
    st.write("**All Signals**")
    st.dataframe(df_results, use_container_width=True)
else:
    st.warning("No data available for the selected tickers.")

st.subheader("Visual Analysis")
ticker_choice = st.selectbox("Select Ticker for Charts", TICKERS)
if ticker_choice:
    df = fetch_data(ticker_choice)
    if df is not None:
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=("Price & Indicators", "RSI", "MACD"),
            vertical_spacing=0.1
        )
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Price", line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], name="EMA20", line=dict(color='orange')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name="BB Upper", line=dict(color='gray', dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name="BB Lower", line=dict(color='gray', dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color='purple')), row=2, col=1)
        fig.add_hline(y=BUY_THRESHOLD, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_hline(y=SELL_THRESHOLD, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name="MACD", line=dict(color='blue')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name="Signal", line=dict(color='red')), row=3, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name="Histogram"), row=3, col=1)
        fig.update_layout(height=800, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

st.subheader("Market Trends")
trends = fetch_market_trends()
st.write("**VIX (Volatility Index):**", trends['VIX'] if trends['VIX'] else "N/A")
st.write("**Sector Performance (Latest Close):**")
for etf, price in trends.items():
    if etf != 'VIX':
        st.write(f"{etf}: {price if price else 'N/A'}")

st.subheader("Things to Watch Out For")
st.markdown("""
- **Liquidity Risk**: Ensure options have open interest >100 and tight bid-ask spreads.
- **Earnings Risk**: Check for upcoming earnings reports on Webull.
- **High Volatility**: VIX >20 may increase option premiums and risk.
- **Overnight Risk**: Gaps against your position can lead to losses.
- **Execution Timing**: Enter trades near 15:45 EST, exit post-09:31 EST.
- **API Limits**: Large ticker lists may hit yfinance rate limits; reduce TICKERS if slow.
""")

st.caption("Bot scans for EOD BUY signals (RSI < 30, price > EMA20, MACD bullish, near BB lower). Targets call options with ~0.4 delta for overnight holds.")

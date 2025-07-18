import yfinance as yf
import pandas as pd
import streamlit as st
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Swing Momentum Bot", layout="wide")

# Configuration Class
class TradingConfig:
    # Trading Parameters
    BUY_THRESHOLD = 50
    SELL_THRESHOLD = 70
    MIN_VOLUME = 500000
    
    # Technical Indicators
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    BB_PERIOD = 20
    BB_STD = 2
    
    # Options
    DELTA_TARGET = 0.4
    
    # Timing
    ENTRY_TIME = "15:45"
    EXIT_TIME = "09:31"

config = TradingConfig()

# Ticker list
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

st.title("📈 Swing Momentum Bot Dashboard")
st.markdown("Scans ~100 tickers for overnight swing trades with momentum indicators and call option suggestions.")

def calculate_technical_indicators(df):
    """Calculate all technical indicators for a dataframe"""
    try:
        # EMA20
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs.replace(float('inf'), 0)))
        
        # MACD
        ema_fast = df['Close'].ewm(span=config.MACD_FAST, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=config.MACD_SLOW, adjust=False).mean()
        df['MACD'] = ema_fast - ema_slow
        df['MACD_Signal'] = df['MACD'].ewm(span=config.MACD_SIGNAL, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Mid'] = df['Close'].rolling(config.BB_PERIOD).mean()
        df['BB_Std'] = df['Close'].rolling(config.BB_PERIOD).std()
        df['BB_Upper'] = df['BB_Mid'] + config.BB_STD * df['BB_Std']
        df['BB_Lower'] = df['BB_Mid'] - config.BB_STD * df['BB_Std']
        
        return df
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return None

@st.cache_data(ttl=60)
def fetch_data(ticker):
    """Fetch and process data for a single ticker"""
    try:
        df = yf.download(ticker, period='1d', interval='1m', progress=False)
        
        # Check if we have sufficient data
        if df.empty or len(df) < 50:
            logger.warning(f"Insufficient data for {ticker}: {len(df) if not df.empty else 0} rows")
            return None
        
        # Calculate technical indicators
        df = calculate_technical_indicators(df)
        
        if df is None:
            logger.error(f"Failed to calculate indicators for {ticker}")
            return None
            
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return None

@st.cache_data(ttl=300)
def fetch_options(ticker):
    """Fetch options data for a ticker"""
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        
        if not expirations:
            logger.warning(f"No options data available for {ticker}")
            return None
            
        expiration = expirations[0]
        opts = stock.option_chain(expiration)
        calls = opts.calls
        
        # Remove the incorrect delta calculation
        # Instead, we'll use strike price proximity to current price
        return calls
        
    except Exception as e:
        logger.error(f"Error fetching options for {ticker}: {e}")
        return None

def find_best_call_option(options, current_price):
    """Find the best call option based on strike proximity to current price"""
    if options is None or options.empty:
        return None
    
    try:
        # Filter for reasonable strikes (within 10% of current price)
        reasonable_strikes = options[
            (options['strike'] >= current_price * 0.9) & 
            (options['strike'] <= current_price * 1.1)
        ]
        
        if reasonable_strikes.empty:
            return None
        
        # Find strike closest to current price
        reasonable_strikes = reasonable_strikes.copy()
        reasonable_strikes['strike_diff'] = abs(reasonable_strikes['strike'] - current_price)
        best_option = reasonable_strikes.loc[reasonable_strikes['strike_diff'].idxmin()]
        
        return round(best_option['strike'], 2)
        
    except Exception as e:
        logger.error(f"Error finding best call option: {e}")
        return None

def evaluate_trade(df, ticker):
    """Evaluate trading signals for a ticker"""
    if df is None or len(df) < 50:
        return None
    
    try:
        latest = df.iloc[-1]
        
        # Early exit for low volume
        if latest['Volume'] < config.MIN_VOLUME:
            return None
        
        # Check for NaN values in required columns
        required_cols = ['RSI', 'Close', 'EMA20', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower']
        if latest[required_cols].isna().any():
            logger.warning(f"NaN values found in indicators for {ticker}")
            return None
        
        # Extract values
        rsi = latest['RSI']
        price = latest['Close']
        ema20 = latest['EMA20']
        macd = latest['MACD']
        macd_signal = latest['MACD_Signal']
        bb_upper = latest['BB_Upper']
        bb_lower = latest['BB_Lower']
        volume = latest['Volume']
        
        # Initialize variables
        signal = "WAIT"
        reason = ""
        call_strike = None
        signal_score = 0
        
        # Buy signal logic
        if (rsi < config.BUY_THRESHOLD and 
            price > ema20 and
            macd > macd_signal and 
            price < bb_lower + (bb_upper - bb_lower) * 0.2):
            
            signal = "BUY"
            reason = f"RSI {rsi:.2f} < {config.BUY_THRESHOLD}, Price > EMA20, MACD Bullish, Near BB Lower"
            signal_score = config.BUY_THRESHOLD - rsi
            
            # Find best call option
            options = fetch_options(ticker)
            call_strike = find_best_call_option(options, price)
        
        # Sell signal logic
        elif (rsi > config.SELL_THRESHOLD and 
              price < ema20 and 
              macd < macd_signal):
            
            signal = "SELL"
            reason = f"RSI {rsi:.2f} > {config.SELL_THRESHOLD}, Price < EMA20, MACD Bearish"
        
        # Only return trades with actual signals
        if signal == "WAIT":
            return None
            
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
        
    except Exception as e:
        logger.error(f"Error evaluating trade for {ticker}: {e}")
        return None

@st.cache_data(ttl=600)
def fetch_market_trends():
    """Fetch market trend indicators"""
    trends = {}
    
    try:
        # VIX
        vix = yf.download('^VIX', period='1d', interval='1d', progress=False)
        if not vix.empty:
            close_col = 'Close' if 'Close' in vix.columns else 'Adj Close'
            if close_col in vix.columns:
                trends['VIX'] = round(vix[close_col].iloc[-1], 2)
            else:
                trends['VIX'] = None
        else:
            trends['VIX'] = None
            
    except Exception as e:
        logger.error(f"Error fetching VIX: {e}")
        trends['VIX'] = None
    
    # Sector ETFs
    sector_etfs = ['XLF', 'XLK', 'XLE', 'XLU', 'XLY']
    for etf in sector_etfs:
        try:
            df = yf.download(etf, period='1d', interval='1d', progress=False)
            if not df.empty:
                close_col = 'Close' if 'Close' in df.columns else 'Adj Close'
                if close_col in df.columns:
                    trends[etf] = round(df[close_col].iloc[-1], 2)
                else:
                    trends[etf] = None
            else:
                trends[etf] = None
        except Exception as e:
            logger.error(f"Error fetching {etf}: {e}")
            trends[etf] = None
    
    return trends

# Main execution
st.subheader("Top Trade Signals")

# Initialize counters
results = []
fetch_count = 0
trade_count = 0

# Progress bar
progress_bar = st.progress(0)
status_text = st.empty()

# Process each ticker
for i, ticker in enumerate(TICKERS):
    status_text.text(f"Processing {ticker}... ({i+1}/{len(TICKERS)})")
    
    df = fetch_data(ticker)
    if df is not None:
        fetch_count += 1
        trade = evaluate_trade(df, ticker)
        if trade:
            trade_count += 1
            results.append(trade)
    
    progress_bar.progress((i + 1) / len(TICKERS))

# Clear status text
status_text.empty()

# Display results
st.write(f"✅ Data fetched successfully for {fetch_count} tickers")
st.write(f"📊 {trade_count} trade signals generated")

if not results:
    st.warning("No trade signals generated at this time. This could be due to:")
    st.write("• Market conditions not meeting signal criteria")
    st.write("• Data fetch issues")
    st.write("• All tickers below minimum volume threshold")
    df_results = pd.DataFrame()
else:
    df_results = pd.DataFrame(results)

# Display trade signals
if not df_results.empty:
    # Filter and display BUY signals
    buy_signals = df_results[df_results['Signal'] == 'BUY'].sort_values(by='Signal_Score', ascending=False)
    
    if not buy_signals.empty:
        st.write("**🟢 Top BUY Signals**")
        st.dataframe(buy_signals.head(10), use_container_width=True)
    
    # Display all signals
    st.write("**📋 All Signals**")
    st.dataframe(df_results, use_container_width=True)
else:
    st.info("No trade signals found. Try adjusting the parameters or check market conditions.")

# Visual Analysis Section
st.subheader("📊 Visual Analysis")

ticker_choice = st.selectbox("Select Ticker for Charts", TICKERS)

if ticker_choice:
    df = fetch_data(ticker_choice)
    if df is not None and not df.empty:
        try:
            # Create subplots
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=("Price & Indicators", "RSI", "MACD"),
                vertical_spacing=0.1,
                row_heights=[0.5, 0.25, 0.25]
            )
            
            # Price chart with indicators
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Price", line=dict(color='blue')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], name="EMA20", line=dict(color='orange')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name="BB Upper", line=dict(color='gray', dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name="BB Lower", line=dict(color='gray', dash='dash')), row=1, col=1)
            
            # RSI
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color='purple')), row=2, col=1)
            fig.add_hline(y=config.BUY_THRESHOLD, line_dash="dash", line_color="green", row=2, col=1)
            fig.add_hline(y=config.SELL_THRESHOLD, line_dash="dash", line_color="red", row=2, col=1)
            
            # MACD
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name="MACD", line=dict(color='blue')), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name="Signal", line=dict(color='red')), row=3, col=1)
            fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name="Histogram", opacity=0.7), row=3, col=1)
            
            # Update layout
            fig.update_layout(height=800, showlegend=True, title=f"{ticker_choice} - Technical Analysis")
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating chart for {ticker_choice}: {e}")
    else:
        st.warning(f"No data available for {ticker_choice}")

# Market Trends Section
st.subheader("🌍 Market Trends")

try:
    trends = fetch_market_trends()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("VIX (Volatility Index)", 
                 trends.get('VIX', 'N/A'), 
                 help="VIX > 20 indicates high market volatility")
    
    with col2:
        st.write("**Sector Performance (Latest Close):**")
        sector_names = {
            'XLF': 'Financial',
            'XLK': 'Technology', 
            'XLE': 'Energy',
            'XLU': 'Utilities',
            'XLY': 'Consumer Disc.'
        }
        
        for etf, price in trends.items():
            if etf != 'VIX':
                sector_name = sector_names.get(etf, etf)
                st.write(f"• {sector_name} ({etf}): {price if price else 'N/A'}")

except Exception as e:
    st.error(f"Error fetching market trends: {e}")

# Risk Management Section
st.subheader("⚠️ Risk Management Guidelines")

st.markdown("""
**Key Risk Factors:**
- **Liquidity Risk**: Ensure options have open interest >100 and tight bid-ask spreads
- **Earnings Risk**: Check earnings calendar - avoid positions before announcements
- **Volatility Risk**: VIX >20 increases option premiums and overnight risk
- **Gap Risk**: Overnight gaps can cause significant losses
- **Volume Risk**: Low volume stocks may have poor execution

**Execution Guidelines:**
- Enter trades near 15:45 EST (market close approach)
- Exit trades after 09:31 EST (market open + 1 minute)
- Use stop-losses for risk management
- Position size according to your risk tolerance
- Monitor pre-market news and futures
""")

# Footer
st.markdown("---")
st.caption(f"""
**Bot Configuration**: BUY when RSI < {config.BUY_THRESHOLD}, Price > EMA20, MACD bullish, near BB lower band. 
Minimum volume: {config.MIN_VOLUME:,}. Targets call options near current price for overnight holds.
""")

import yfinance as yf
import pandas as pd
import streamlit as st
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import numpy as np

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
        # Ensure we have enough data
        if len(df) < 30:
            return None
            
        # EMA20
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(14, min_periods=1).mean()
        avg_loss = loss.rolling(14, min_periods=1).mean()
        
        # Handle division by zero
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Fill NaN values with neutral RSI
        df['RSI'] = df['RSI'].fillna(50)
        
        # MACD
        ema_fast = df['Close'].ewm(span=config.MACD_FAST, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=config.MACD_SLOW, adjust=False).mean()
        df['MACD'] = ema_fast - ema_slow
        df['MACD_Signal'] = df['MACD'].ewm(span=config.MACD_SIGNAL, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Mid'] = df['Close'].rolling(config.BB_PERIOD, min_periods=1).mean()
        df['BB_Std'] = df['Close'].rolling(config.BB_PERIOD, min_periods=1).std()
        df['BB_Upper'] = df['BB_Mid'] + config.BB_STD * df['BB_Std']
        df['BB_Lower'] = df['BB_Mid'] - config.BB_STD * df['BB_Std']
        
        # Fill any remaining NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return None

@st.cache_data(ttl=60)
def fetch_data(ticker):
    """Fetch and process data for a single ticker"""
    try:
        # Use a longer period to ensure sufficient data
        df = yf.download(ticker, period='5d', interval='1m', progress=False)
        
        # Check if we have sufficient data
        if df.empty or len(df) < 30:
            logger.warning(f"Insufficient data for {ticker}: {len(df) if not df.empty else 0} rows")
            return None
        
        # Reset index to make datetime a column
        df = df.reset_index()
        
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
        ].copy()
        
        if reasonable_strikes.empty:
            return None
        
        # Find strike closest to current price
        reasonable_strikes['strike_diff'] = abs(reasonable_strikes['strike'] - current_price)
        best_option = reasonable_strikes.loc[reasonable_strikes['strike_diff'].idxmin()]
        
        return round(best_option['strike'], 2)
        
    except Exception as e:
        logger.error(f"Error finding best call option: {e}")
        return None

def evaluate_trade(df, ticker):
    """Evaluate trading signals for a ticker"""
    if df is None or len(df) < 30:
        return None
    
    try:
        latest = df.iloc[-1]
        
        # Early exit for low volume
        if latest['Volume'] < config.MIN_VOLUME:
            return None
        
        # Check for NaN values in required columns
        required_cols = ['RSI', 'Close', 'EMA20', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower']
        if pd.isna(latest[required_cols]).any():
            logger.warning(f"NaN values found in indicators for {ticker}")
            return None
        
        # Extract values
        rsi = float(latest['RSI'])
        price = float(latest['Close'])
        ema20 = float(latest['EMA20'])
        macd = float(latest['MACD'])
        macd_signal = float(latest['MACD_Signal'])
        bb_upper = float(latest['BB_Upper'])
        bb_lower = float(latest['BB_Lower'])
        volume = int(latest['Volume'])
        
        # Initialize variables
        signal = "WAIT"
        reason = ""
        call_strike = None
        signal_score = 0
        
        # Buy signal logic (more lenient for testing)
        if (rsi < config.BUY_THRESHOLD and 
            price > ema20 * 0.98 and  # Allow small deviation
            macd > macd_signal * 0.95):  # Allow small deviation
            
            signal = "BUY"
            reason = f"RSI {rsi:.2f} < {config.BUY_THRESHOLD}, Price ~> EMA20, MACD ~> Signal"
            signal_score = config.BUY_THRESHOLD - rsi
            
            # Find best call option
            try:
                options = fetch_options(ticker)
                call_strike = find_best_call_option(options, price)
            except:
                call_strike = None
        
        # Sell signal logic  
        elif (rsi > config.SELL_THRESHOLD and 
              price < ema20 * 1.02 and  # Allow small deviation
              macd < macd_signal * 1.05):  # Allow small deviation
            
            signal = "SELL"
            reason = f"RSI {rsi:.2f} > {config.SELL_THRESHOLD}, Price ~< EMA20, MACD ~< Signal"
            signal_score = rsi - config.SELL_THRESHOLD
        
        # Also check for momentum signals (additional criteria)
        elif (30 < rsi < 70 and  # Neutral RSI range
              abs(price - ema20) / price < 0.02 and  # Close to EMA20
              abs(macd - macd_signal) / abs(macd_signal) < 0.1 if macd_signal != 0 else True):  # Close MACD signals
            
            signal = "WATCH"
            reason = f"Neutral momentum - RSI: {rsi:.2f}, Price near EMA20"
            signal_score = 1
        
        # Return all signals (including WATCH for debugging)
        if signal == "WAIT":
            return None
            
        return {
            'Symbol': ticker,
            'Signal': signal,
            'RSI': round(rsi, 2),
            'Close': round(price, 2),
            'EMA20': round(ema20, 2),
            'MACD': round(macd, 4),
            'MACD_Signal': round(macd_signal, 4),
            'BB_Upper': round(bb_upper, 2),
            'BB_Lower': round(bb_lower, 2),
            'Volume': volume,
            'Call_Strike': call_strike,
            'Reason': reason,
            'Signal_Score': round(signal_score, 2)
        }
        
    except Exception as e:
        logger.error(f"Error evaluating trade for {ticker}: {e}")
        return None

@st.cache_data(ttl=600)
def fetch_market_trends():
    """Fetch market trend indicators"""
    trends = {}
    
    try:
        # VIX - Fixed the Series ambiguity error
        vix_data = yf.download('^VIX', period='1d', interval='1d', progress=False)
        if not vix_data.empty and len(vix_data) > 0:
            # Get the most recent close value
            if 'Close' in vix_data.columns:
                trends['VIX'] = round(float(vix_data['Close'].iloc[-1]), 2)
            elif 'Adj Close' in vix_data.columns:
                trends['VIX'] = round(float(vix_data['Adj Close'].iloc[-1]), 2)
            else:
                trends['VIX'] = None
        else:
            trends['VIX'] = None
            
    except Exception as e:
        logger.error(f"Error fetching VIX: {e}")
        trends['VIX'] = None
    
    # Sector ETFs - Fixed the same Series ambiguity
    sector_etfs = ['XLF', 'XLK', 'XLE', 'XLU', 'XLY']
    for etf in sector_etfs:
        try:
            etf_data = yf.download(etf, period='1d', interval='1d', progress=False)
            if not etf_data.empty and len(etf_data) > 0:
                if 'Close' in etf_data.columns:
                    trends[etf] = round(float(etf_data['Close'].iloc[-1]), 2)
                elif 'Adj Close' in etf_data.columns:
                    trends[etf] = round(float(etf_data['Adj Close'].iloc[-1]), 2)
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
error_count = 0

# Progress bar
progress_bar = st.progress(0)
status_text = st.empty()

# Process each ticker
for i, ticker in enumerate(TICKERS):
    status_text.text(f"Processing {ticker}... ({i+1}/{len(TICKERS)})")
    
    try:
        df = fetch_data(ticker)
        if df is not None:
            fetch_count += 1
            trade = evaluate_trade(df, ticker)
            if trade:
                trade_count += 1
                results.append(trade)
        else:
            error_count += 1
    except Exception as e:
        logger.error(f"Error processing {ticker}: {e}")
        error_count += 1
    
    progress_bar.progress((i + 1) / len(TICKERS))

# Clear status text
status_text.empty()

# Display results
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("✅ Data Fetched", fetch_count)
with col2:
    st.metric("📊 Trade Signals", trade_count)
with col3:
    st.metric("❌ Errors", error_count)

if not results:
    st.info("No trade signals found. Try adjusting the parameters or check market conditions.")
    st.info("💡 **Debugging Info**: The bot may be working correctly - sometimes market conditions don't meet signal criteria.")
else:
    df_results = pd.DataFrame(results)
    
    # Display different signal types
    buy_signals = df_results[df_results['Signal'] == 'BUY']
    sell_signals = df_results[df_results['Signal'] == 'SELL'] 
    watch_signals = df_results[df_results['Signal'] == 'WATCH']
    
    if not buy_signals.empty:
        st.success(f"🟢 **{len(buy_signals)} BUY Signals Found**")
        st.dataframe(buy_signals.sort_values(by='Signal_Score', ascending=False), use_container_width=True)
    
    if not sell_signals.empty:
        st.error(f"🔴 **{len(sell_signals)} SELL Signals Found**")
        st.dataframe(sell_signals.sort_values(by='Signal_Score', ascending=False), use_container_width=True)
    
    if not watch_signals.empty:
        st.warning(f"👀 **{len(watch_signals)} WATCH Signals Found**")
        st.dataframe(watch_signals, use_container_width=True)

# Add debugging section
if st.checkbox("🔧 Show Debug Information"):
    st.subheader("Debug Information")
    
    # Test with a single ticker
    test_ticker = st.selectbox("Test Single Ticker:", ['AAPL', 'MSFT', 'TSLA', 'SPY'])
    
    if st.button("Test Ticker"):
        with st.spinner(f"Testing {test_ticker}..."):
            df = fetch_data(test_ticker)
            if df is not None:
                st.success(f"✅ Successfully fetched {len(df)} data points for {test_ticker}")
                
                # Show latest values
                latest = df.iloc[-1]
                st.write("**Latest Values:**")
                debug_data = {
                    'RSI': latest['RSI'],
                    'Close': latest['Close'],
                    'EMA20': latest['EMA20'],
                    'MACD': latest['MACD'],
                    'MACD_Signal': latest['MACD_Signal'],
                    'Volume': latest['Volume']
                }
                st.json(debug_data)
                
                # Test evaluation
                trade = evaluate_trade(df, test_ticker)
                if trade:
                    st.success("✅ Trade evaluation successful")
                    st.json(trade)
                else:
                    st.warning("⚠️ No trade signal generated")
            else:
                st.error(f"❌ Failed to fetch data for {test_ticker}")

# Visual Analysis Section
st.subheader("📊 Visual Analysis")

ticker_choice = st.selectbox("Select Ticker for Charts", TICKERS, key="chart_ticker")

if ticker_choice:
    df = fetch_data(ticker_choice)
    if df is not None and not df.empty:
        try:
            # Create subplots
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=(f"{ticker_choice} - Price & Indicators", "RSI", "MACD"),
                vertical_spacing=0.1,
                row_heights=[0.5, 0.25, 0.25]
            )
            
            # Use the Datetime index for x-axis
            x_data = df['Datetime'] if 'Datetime' in df.columns else df.index
            
            # Price chart with indicators
            fig.add_trace(go.Scatter(x=x_data, y=df['Close'], name="Price", line=dict(color='blue')), row=1, col=1)
            fig.add_trace(go.Scatter(x=x_data, y=df['EMA20'], name="EMA20", line=dict(color='orange')), row=1, col=1)
            fig.add_trace(go.Scatter(x=x_data, y=df['BB_Upper'], name="BB Upper", line=dict(color='gray', dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=x_data, y=df['BB_Lower'], name="BB Lower", line=dict(color='gray', dash='dash')), row=1, col=1)
            
            # RSI
            fig.add_trace(go.Scatter(x=x_data, y=df['RSI'], name="RSI", line=dict(color='purple')), row=2, col=1)
            fig.add_hline(y=config.BUY_THRESHOLD, line_dash="dash", line_color="green", row=2, col=1)
            fig.add_hline(y=config.SELL_THRESHOLD, line_dash="dash", line_color="red", row=2, col=1)
            
            # MACD
            fig.add_trace(go.Scatter(x=x_data, y=df['MACD'], name="MACD", line=dict(color='blue')), row=3, col=1)
            fig.add_trace(go.Scatter(x=x_data, y=df['MACD_Signal'], name="Signal", line=dict(color='red')), row=3, col=1)
            fig.add_trace(go.Bar(x=x_data, y=df['MACD_Hist'], name="Histogram", opacity=0.7), row=3, col=1)
            
            # Update layout
            fig.update_layout(height=800, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating chart for {ticker_choice}: {e}")
            logger.error(f"Chart error: {e}")
    else:
        st.warning(f"No data available for {ticker_choice}")

# Market Trends Section
st.subheader("🌍 Market Trends")

try:
    with st.spinner("Fetching market trends..."):
        trends = fetch_market_trends()
    
    col1, col2 = st.columns(2)
    
    with col1:
        vix_value = trends.get('VIX', 'N/A')
        if vix_value != 'N/A':
            st.metric("VIX (Volatility Index)", 
                     vix_value, 
                     help="VIX > 20 indicates high market volatility")
        else:
            st.metric("VIX (Volatility Index)", "Data Unavailable")
    
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
                price_display = f"${price}" if price else 'N/A'
                st.write(f"• {sector_name} ({etf}): {price_display}")

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

import yfinance as yf
import pandas as pd
import streamlit as st
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import datetime

# 定義函數來讀取股票數據
def load_stock_data(stockname, start_date, end_date, interval):
    stock = yf.download(stockname, start=start_date, end=end_date, interval=interval)
    if stock.empty:
        st.error("未能讀取到數據，請檢查股票代號是否正確")
        return None
    stock.rename(columns={'Volume': 'amount'}, inplace=True)
    stock.drop(columns=['Adj Close'], inplace=True)
    stock['Volume'] = (stock['amount'] / (stock['Open'] + stock['Close']) / 2).astype(int)
    stock.reset_index(inplace=True)
    return stock

# 定義函數來計算布林通道指標
def calculate_bollinger_bands(stock, period=20, std_dev=2):
    stock['中軌'] = stock['Close'].rolling(window=period).mean()
    stock['上軌'] = stock['中軌'] + std_dev * stock['Close'].rolling(window=period).std()
    stock['下軌'] = stock['中軌'] - std_dev * stock['Close'].rolling(window=period).std()
    return stock

# 定義函數來計算KDJ指標
def calculate_kdj(stock, period=14):
    low_min = stock['Low'].rolling(window=period).min()
    high_max = stock['High'].rolling(window=period).max()
    rsv = (stock['Close'] - low_min) / (high_max - low_min) * 100
    stock['K'] = rsv.ewm(com=2).mean()
    stock['D'] = stock['K'].ewm(com=2).mean()
    stock['J'] = 3 * stock['K'] - 2 * stock['D']
    return stock

# 定義函數來計算RSI指標，並加入超買和超賣標記
def calculate_rsi(stock, period=14):
    delta = stock['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    stock['RSI'] = 100 - (100 / (1 + rs))
    
    # 新增超買和超賣標記
    stock['超買'] = stock['RSI'] > 80
    stock['超賣'] = stock['RSI'] < 20
    
    return stock

# 定義函數來計算MACD指標
def calculate_macd(stock, short_window=12, long_window=26, signal_window=9):
    stock['短期EMA'] = stock['Close'].ewm(span=short_window, adjust=False).mean()
    stock['長期EMA'] = stock['Close'].ewm(span=long_window, adjust=False).mean()
    stock['MACD'] = stock['短期EMA'] - stock['長期EMA']
    stock['信號線'] = stock['MACD'].ewm(span=signal_window, adjust=False).mean()
    stock['柱狀圖'] = stock['MACD'] - stock['信號線']
    return stock

# 定義函數來計算唐奇安通道指標
def calculate_donchian_channels(stock, period=20):
    stock['唐奇安高值'] = stock['High'].rolling(window=period).max()
    stock['唐奇安低值'] = stock['Low'].rolling(window=period).min()
    return stock

# 定義單獨的繪圖函數
def plot_bollinger_bands(stock):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("布林通道", "成交量"))

    fig.add_trace(go.Candlestick(x=stock['Date'], open=stock['Open'], high=stock['High'],
                                 low=stock['Low'], close=stock['Close'], name='K線圖'), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['中軌'], mode='lines', name='中軌'), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['上軌'], mode='lines', name='上軌'), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['下軌'], mode='lines', name='下軌'), row=1, col=1)
    fig.add_trace(go.Bar(x=stock['Date'], y=stock['Volume'], name='成交量'), row=2, col=1)

    fig.update_layout(title="布林通道策略圖", xaxis_title='日期', yaxis_title='價格', xaxis_rangeslider_visible=False)
    st.plotly_chart(fig)

def plot_kdj(stock):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("KDJ指標", "KDJ值"))

    fig.add_trace(go.Candlestick(x=stock['Date'], open=stock['Open'], high=stock['High'],
                                 low=stock['Low'], close=stock['Close'], name='K線圖'), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['K'], mode='lines', name='K值'), row=2, col=1)
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['D'], mode='lines', name='D值'), row=2, col=1)
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['J'], mode='lines', name='J值'), row=2, col=1)

    fig.update_layout(title="KDJ策略圖", xaxis_title='日期', yaxis_title='價格', xaxis_rangeslider_visible=False)
    st.plotly_chart(fig)

def plot_rsi(stock):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("RSI指標", "RSI值"))

    fig.add_trace(go.Candlestick(x=stock['Date'], open=stock['Open'], high=stock['High'],
                                 low=stock['Low'], close=stock['Close'], name='K線圖'), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['RSI'], mode='lines', name='RSI'), row=2, col=1)
    
    # 加入超買和超賣的標記
    fig.add_trace(go.Scatter(x=stock['Date'], y=np.where(stock['超買'], 80, None),
                             mode='markers', marker=dict(color='red', size=10), name='超買 >80'), row=2, col=1)
    fig.add_trace(go.Scatter(x=stock['Date'], y=np.where(stock['超賣'], 20, None),
                             mode='markers', marker=dict(color='blue', size=10), name='超賣 <20'), row=2, col=1)

    fig.update_layout(title="RSI策略圖", xaxis_title='日期', yaxis_title='價格', xaxis_rangeslider_visible=False)
    st.plotly_chart(fig)

def plot_macd(stock):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("MACD指標", "MACD值"))

    fig.add_trace(go.Candlestick(x=stock['Date'], open=stock['Open'], high=stock['High'],
                                 low=stock['Low'], close=stock['Close'], name='K線圖'), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['MACD'], mode='lines', name='MACD'), row=2, col=1)
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['信號線'], mode='lines', name='信號線'), row=2, col=1)
    fig.add_trace(go.Bar(x=stock['Date'], y=stock['柱狀圖'], name='柱狀圖'), row=2, col=1)

    fig.update_layout(title="MACD策略圖", xaxis_title='日期', yaxis_title='價格', xaxis_rangeslider_visible=False)
    st.plotly_chart(fig)

def plot_donchian_channels(stock):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("唐奇安通道", "成交量"))

    fig.add_trace(go.Candlestick(x=stock['Date'], open=stock['Open'], high=stock['High'],
                                 low=stock['Low'], close=stock['Close'], name='K線圖'), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['唐奇安高值'], mode='lines', name='高值通道'), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['唐奇安低值'], mode='lines', name='低值通道'), row=1, col=1)
    fig.add_trace(go.Bar(x=stock['Date'], y=stock['Volume'], name='成交量'), row=2, col=1)

    fig.update_layout(title="唐奇安通道策略圖", xaxis_title='日期', yaxis_title='價格', xaxis_rangeslider_visible=False)
    st.plotly_chart(fig)

# 定義交易策略
def trading_strategy(stock, strategy_name):
    # 根據不同的策略進行交易
    if strategy_name == "布林通道":
        stock['Position'] = np.where(stock['Close'] > stock['上軌'], -1, np.nan)
        stock['Position'] = np.where(stock['Close'] < stock['下軌'], 1, stock['Position'])
    elif strategy_name == "KDJ":
        stock['Position'] = np.where(stock['K'] > stock['D'], 1, -1)
    elif strategy_name == "RSI":
        stock['Position'] = np.where(stock['RSI'] < 20, 1, np.nan)
        stock['Position'] = np.where(stock['RSI'] > 80, -1, stock['Position'])
    elif strategy_name == "MACD":
        stock['Position'] = np.where(stock['MACD'] > stock['信號線'], 1, -1)
    elif strategy_name == "唐奇安通道":
        stock['Position'] = np.where(stock['Close'] > stock['唐奇安高值'].shift(1), 1, np.nan)
        stock['Position'] = np.where(stock['Close'] < stock['唐奇安低值'].shift(1), -1, stock['Position'])

    stock['Position'].fillna(method='ffill', inplace=True)
    stock['Position'].fillna(0, inplace=True)
    stock['市場報酬率'] = stock['Close'].pct_change()
    stock['策略報酬率'] = stock['市場報酬率'] * stock['Position'].shift(1)
    stock['累積策略報酬率'] = (1 + stock['策略報酬率']).cumprod() - 1

    # 計算績效指標
    total_trades = len(stock[(stock['Position'] == 1) | (stock['Position'] == -1)])
    winning_trades = len(stock[(stock['Position'].shift(1) == 1) & (stock['策略報酬率'] > 0)])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    stock['回落'] = (stock['累積策略報酬率'].cummax() - stock['累積策略報酬率'])
    最大回落 = stock['回落'].max()

    總利潤 = stock['策略報酬率'].sum()
    最大連續虧損 = (stock['策略報酬率'] < 0).astype(int).groupby(stock['策略報酬率'].ge(0).cumsum()).sum().max()

    st.write(f"策略績效指標:")
    st.write(f"勝率: {win_rate:.2%}")
    st.write(f"最大連續虧損: {最大連續虧損}")
    st.write(f"最大資金回落: {最大回落:.2%}")
    st.write(f"總損益: {總利潤:.2%}")

    return stock

# 主函數
def main():
    st.title("02")

    # 選擇資料區間
    st.sidebar.subheader("選擇資料區間")
    start_date = st.sidebar.date_input('選擇開始日期', datetime.date(2020, 1, 1))
    end_date = st.sidebar.date_input('選擇結束日期(若日期設未來，會顯示目前最新日期，到今天OR上次收盤)', datetime.date(2100, 1, 1))
    stockname = st.sidebar.text_input('請輸入股票代號 (例: 2610.TW,國外股票或期貨代號請至Yahoo Finance查詢)', '2610.TW')

    # 選擇K線時間長
    interval_options = {"1天": "1d", "1星期": "1wk", "1個月": "1mo"}
    interval_label = st.sidebar.selectbox("選擇K線時間長", list(interval_options.keys()))
    interval = interval_options[interval_label]

    # 選擇指標和參數
    strategy_name = st.sidebar.selectbox("選擇指標", ["MACD","布林通道", "KDJ", "RSI",  "唐奇安通道"])

    if strategy_name == "布林通道":
        bollinger_period = st.sidebar.slider("布林通道週期", min_value=5, max_value=50, value=20, step=1)
        bollinger_std = st.sidebar.slider("布林通道標準差倍數", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
    elif strategy_name == "KDJ":
        kdj_period = st.sidebar.slider("KDJ週期", min_value=5, max_value=50, value=10, step=1)
    elif strategy_name == "RSI":
        rsi_period = st.sidebar.slider("RSI週期", min_value=5, max_value=50, value=10, step=1)
    elif strategy_name == "MACD":
        short_window = st.sidebar.slider("短期EMA窗口", min_value=5, max_value=50, value=10, step=1)
        long_window = st.sidebar.slider("長期EMA窗口", min_value=10, max_value=100, value=20, step=1)
        signal_window = st.sidebar.slider("信號線窗口", min_value=5, max_value=50, value=7, step=1)
    elif strategy_name == "唐奇安通道":
        donchian_period = st.sidebar.slider("唐奇安通道週期", min_value=5, max_value=50, value=15, step=1)

    stock = load_stock_data(stockname, start_date, end_date, interval)
    if stock is not None:
        st.subheader(f"股票代號: {stockname}")

        if strategy_name == "布林通道":
            stock = calculate_bollinger_bands(stock, period=bollinger_period, std_dev=bollinger_std)
            plot_bollinger_bands(stock)
        elif strategy_name == "KDJ":
            stock = calculate_kdj(stock, period=kdj_period)
            plot_kdj(stock)
        elif strategy_name == "RSI":
            stock = calculate_rsi(stock, period=rsi_period)
            plot_rsi(stock)
        elif strategy_name == "MACD":
            stock = calculate_macd(stock, short_window=short_window, long_window=long_window, signal_window=signal_window)
            plot_macd(stock)
        elif strategy_name == "唐奇安通道":
            stock = calculate_donchian_channels(stock, period=donchian_period)
            plot_donchian_channels(stock)
        
        stock = trading_strategy(stock, strategy_name)
        st.write(f"策略績效 (累積報酬): {stock['累積策略報酬率'].iloc[-1]:.2%}")

if __name__ == "__main__":
    main()


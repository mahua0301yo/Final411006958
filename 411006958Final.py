import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import datetime
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# ======================== 資料讀取 ========================
def load_stock_data(stockname, start_date, end_date, interval):
    stock = yf.download(stockname, start=start_date, end=end_date, interval=interval)
    if stock.empty:
        st.error("未能讀取到數據，請檢查股票代號是否正確")
        return None

    stock.rename(columns={'Volume': 'amount'}, inplace=True)
    stock.drop(columns=['Adj Close'], inplace=True, errors='ignore')

    stock['Volume'] = ((stock['amount'] / (stock['Open'] + stock['Close']) / 2).fillna(0)).astype(int)
    stock.reset_index(inplace=True)
    return stock

# ======================== 技術指標計算 ========================
def calculate_bollinger_bands(stock, period=20, std_dev=2):
    stock['中軌'] = stock['Close'].rolling(window=period).mean()
    stock['上軌'] = stock['中軌'] + std_dev * stock['Close'].rolling(window=period).std()
    stock['下軌'] = stock['中軌'] - std_dev * stock['Close'].rolling(window=period).std()
    return stock

def calculate_kdj(stock, period=14):
    low_min = stock['Low'].rolling(window=period).min()
    high_max = stock['High'].rolling(window=period).max()
    rsv = (stock['Close'] - low_min) / (high_max - low_min) * 100
    stock['K'] = rsv.ewm(com=2).mean()
    stock['D'] = stock['K'].ewm(com=2).mean()
    stock['J'] = 3 * stock['K'] - 2 * stock['D']
    return stock

def calculate_rsi(stock, period=14):
    delta = stock['Close'].diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss
    stock['RSI'] = 100 - (100 / (1 + rs))
    stock['超買'] = stock['RSI'] > 80
    stock['超賣'] = stock['RSI'] < 20
    return stock

def calculate_macd(stock, short_window=12, long_window=26, signal_window=9):
    stock['短期EMA'] = stock['Close'].ewm(span=short_window, adjust=False).mean()
    stock['長期EMA'] = stock['Close'].ewm(span=long_window, adjust=False).mean()
    stock['MACD'] = stock['短期EMA'] - stock['長期EMA']
    stock['信號線'] = stock['MACD'].ewm(span=signal_window, adjust=False).mean()
    stock['柱狀圖'] = stock['MACD'] - stock['信號線']
    return stock

def calculate_donchian_channels(stock, period=20):
    stock['唐奇安高值'] = stock['High'].rolling(window=period).max()
    stock['唐奇安低值'] = stock['Low'].rolling(window=period).min()
    return stock

# ======================== 技術指標繪圖 ========================
def plot_candlestick_with_indicators(stock, indicators=[], title="技術指標圖"):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)

    # K 線圖
    fig.add_trace(go.Candlestick(
        x=stock['Date'], open=stock['Open'], high=stock['High'],
        low=stock['Low'], close=stock['Close'], name='K線圖'), row=1, col=1)

    # 指標線
    for indicator in indicators:
        fig.add_trace(go.Scatter(
            x=stock['Date'], y=stock[indicator['column']],
            mode=indicator.get('mode', 'lines'), name=indicator['name'],
            marker=indicator.get('marker', None)
        ), row=indicator.get('row', 1), col=1)

    # 成交量
    fig.add_trace(go.Bar(
        x=stock['Date'], y=stock['Volume'], name='成交量'), row=2, col=1)

    fig.update_layout(
        title=title,
        xaxis_title='日期',
        yaxis_title='價格',
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig)

# ======================== 交易策略 ========================
def trading_strategy(stock, strategy_name):
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

    stock['Position'].ffill(inplace=True)
    stock['Position'].fillna(0, inplace=True)

    stock['市場報酬率'] = stock['Close'].pct_change()
    stock['策略報酬率'] = stock['市場報酬率'] * stock['Position'].shift(1)
    stock['累積策略報酬率'] = (1 + stock['策略報酬率']).cumprod() - 1

    # 績效統計
    total_trades = len(stock[(stock['Position'] == 1) | (stock['Position'] == -1)])
    winning_trades = len(stock[(stock['Position'].shift(1) == 1) & (stock['策略報酬率'] > 0)])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    stock['回落'] = stock['累積策略報酬率'].cummax() - stock['累積策略報酬率']
    最大回落 = stock['回落'].max()

    總利潤 = stock['策略報酬率'].sum()
    最大連續虧損 = (stock['策略報酬率'] < 0).astype(int).groupby(stock['策略報酬率'].ge(0).cumsum()).sum().max()

    # 顯示績效
    st.write("## 策略績效指標")
    st.write(f"勝率: {win_rate:.2%}")
    st.write(f"最大連續虧損: {最大連續虧損}")
    st.write(f"最大資金回落: {最大回落:.2%}")
    st.write(f"總損益: {總利潤:.2%}")

    return stock

# ======================== 主程式 ========================
def main():
    st.title("股票技術指標預測系統")

    # Sidebar 選項
    st.sidebar.header("設定參數")
    start_date = st.sidebar.date_input("開始日期", datetime.date(2020, 1, 1))
    end_date = st.sidebar.date_input("結束日期", datetime.date.today())
    stockname = st.sidebar.text_input("輸入股票代號 (如 2610.TW)", "2610.TW")

    interval_options = {"1天": "1d", "1週": "1wk", "1月": "1mo"}
    interval_label = st.sidebar.selectbox("選擇時間範圍", list(interval_options.keys()))
    interval = interval_options[interval_label]

    strategy_name = st.sidebar.selectbox("選擇技術指標", ["MACD", "布林通道", "KDJ", "RSI", "唐奇安通道"])

    # 策略參數
    if strategy_name == "布林通道":
        bollinger_period = st.sidebar.slider("週期", 5, 50, 20)
        bollinger_std = st.sidebar.slider("標準差倍數", 1.0, 3.0, 2.0, 0.1)
    elif strategy_name == "KDJ":
        kdj_period = st.sidebar.slider("週期", 5, 50, 14)
    elif strategy_name == "RSI":
        rsi_period = st.sidebar.slider("週期", 5, 50, 14)
    elif strategy_name == "MACD":
        short_window = st.sidebar.slider("短期EMA", 5, 50, 12)
        long_window = st.sidebar.slider("長期EMA", 10, 100, 26)
        signal_window = st.sidebar.slider("信號線EMA", 5, 50, 9)
    elif strategy_name == "唐奇安通道":
        donchian_period = st.sidebar.slider("週期", 5, 50, 20)

    # 載入資料
    stock = load_stock_data(stockname, start_date, end_date, interval)
    if stock is None:
        return

    # 計算指標
    if strategy_name == "布林通道":
        stock = calculate_bollinger_bands(stock, bollinger_period, bollinger_std)
        indicators = [
            {'column': '中軌', 'name': '中軌'},
            {'column': '上軌', 'name': '上軌'},
            {'column': '下軌', 'name': '下軌'}
        ]
    elif strategy_name == "KDJ":
        stock = calculate_kdj(stock, kdj_period)
        indicators = [
            {'column': 'K', 'name': 'K值', 'row': 2},
            {'column': 'D', 'name': 'D值', 'row': 2},
            {'column': 'J', 'name': 'J值', 'row': 2}
        ]
    elif strategy_name == "RSI":
        stock = calculate_rsi(stock, rsi_period)
        indicators = [
            {'column': 'RSI', 'name': 'RSI', 'row': 2},
            {'column': '超買', 'name': '超買 >80', 'mode': 'markers', 'row': 2,
             'marker': dict(color='red', size=8)},
            {'column': '超賣', 'name': '超賣 <20', 'mode': 'markers', 'row': 2,
             'marker': dict(color='blue', size=8)}
        ]
    elif strategy_name == "MACD":
        stock = calculate_macd(stock, short_window, long_window, signal_window)
        indicators = [
            {'column': 'MACD', 'name': 'MACD', 'row': 2},
            {'column': '信號線', 'name': '信號線', 'row': 2}
        ]
    elif strategy_name == "唐奇安通道":
        stock = calculate_donchian_channels(stock, donchian_period)
        indicators = [
            {'column': '唐奇安高值', 'name': '高值通道'},
            {'column': '唐奇安低值', 'name': '低值通道'}
        ]

    # 畫圖
    plot_candlestick_with_indicators(stock, indicators, title=f"{strategy_name} 策略圖")

    # 策略回測
    stock = trading_strategy(stock, strategy_name)
    st.write(f"## 策略累積報酬率：{stock['累積策略報酬率'].iloc[-1]:.2%}")

if __name__ == "__main__":
    main()

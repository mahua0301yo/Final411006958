import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

# ------------------ 資料讀取與預處理 ------------------ #
def load_stock_data(stockname, start_date, end_date, interval):
    stock = yf.download(stockname, start=start_date, end=end_date, interval=interval)
    if stock.empty:
        st.error("未能讀取到數據，請檢查股票代號或日期區間")
        return None

    # 安全改名
    if 'Volume' in stock.columns:
        stock.rename(columns={'Volume': 'amount'}, inplace=True)

    # 安全刪除
    if 'Adj Close' in stock.columns:
        stock.drop(columns=['Adj Close'], inplace=True)

    # 重算成交量
    if {'amount', 'Open', 'Close'}.issubset(stock.columns):
        stock['Volume'] = ((stock['amount'] / (stock['Open'] + stock['Close']) / 2).fillna(0)).astype(int)
    else:
        stock['Volume'] = 0  # 預設為 0

    stock.reset_index(inplace=True)
    return stock

# ------------------ 指標計算 ------------------ #
def calculate_indicators(df):
    df['SMA5'] = df['Close'].rolling(window=5).mean()
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['UpperBand'] = df['Close'].rolling(window=20).mean() + 2 * df['Close'].rolling(window=20).std()
    df['LowerBand'] = df['Close'].rolling(window=20).mean() - 2 * df['Close'].rolling(window=20).std()
    return df

# ------------------ 畫圖 ------------------ #
def plot_candlestick_with_indicators(df, title="K線圖"):
    fig = go.Figure()

    # K線
    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='K線'
    ))

    # 均線
    if 'SMA5' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['SMA5'],
            line=dict(color='blue', width=1),
            name='SMA5'
        ))

    if 'SMA20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['SMA20'],
            line=dict(color='orange', width=1),
            name='SMA20'
        ))

    # 布林通道
    if 'UpperBand' in df.columns and 'LowerBand' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['UpperBand'],
            line=dict(color='green', width=1, dash='dot'),
            name='Upper Band'
        ))
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['LowerBand'],
            line=dict(color='red', width=1, dash='dot'),
            name='Lower Band'
        ))

    fig.update_layout(
        title=title,
        xaxis_title="日期",
        yaxis_title="價格",
        xaxis_rangeslider_visible=False,
        width=1000,
        height=600
    )

    st.plotly_chart(fig)

# ------------------ Streamlit 介面 ------------------ #
def main():
    st.title("股票預測平台")

    stockname = st.text_input("輸入股票代碼 (如: AAPL、2330.TW)", "AAPL")
    start_date = st.date_input("選擇起始日期", pd.to_datetime("2023-01-01"))
    end_date = st.date_input("選擇結束日期", pd.to_datetime("today"))
    interval = st.selectbox("選擇時間區間", ("1d", "1wk", "1mo"))

    if st.button("開始分析"):
        with st.spinner('載入中...'):
            stock = load_stock_data(stockname, start_date, end_date, interval)

        if stock is not None:
            stock = calculate_indicators(stock)
            st.success(f"{stockname} 資料讀取與指標計算完成")
            plot_candlestick_with_indicators(stock, title=f"{stockname} K線及指標圖")

            st.subheader("原始數據")
            st.dataframe(stock)
        else:
            st.warning("無法載入資料，請確認輸入正確。")

if __name__ == "__main__":
    main()

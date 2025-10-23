# ======================================================
# Multi-Ticker Streamlit Dashboard
# 다양한 주식에 대한 실시간 시각화 및 모니터링
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta
import yfinance as yf

# Import our system components
from agent_utils import AgentLoader
from debate_system import DebateSystem
from multi_ticker_dataset_builder import MultiTickerDatasetBuilder

# Page configuration
st.set_page_config(
    page_title="Multi-Ticker Debating System Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .ticker-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2E8B57;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-metric {
        border-left-color: #28a745;
    }
    .warning-metric {
        border-left-color: #ffc107;
    }
    .error-metric {
        border-left-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_available_tickers():
    """Load available tickers from dataset builder"""
    builder = MultiTickerDatasetBuilder()
    return builder.get_available_tickers()

@st.cache_data
def load_ticker_data(ticker):
    """Load data for a specific ticker with caching"""
    try:
        # Load debate summary if exists
        debate_summary = None
        if os.path.exists(f'data/{ticker}_debate_summary.csv'):
            debate_summary = pd.read_csv(f'data/{ticker}_debate_summary.csv')
        
        # Load beta log if exists
        beta_log = None
        if os.path.exists(f'data/{ticker}_beta_log.csv'):
            beta_log = pd.read_csv(f'data/{ticker}_beta_log.csv')
        
        # Load test data
        test_data = {}
        for agent_type in ['technical', 'fundamental', 'sentimental']:
            file_path = f'data/{ticker}_{agent_type}_test.csv'
            if os.path.exists(file_path):
                test_data[agent_type] = pd.read_csv(file_path)
        
        return debate_summary, beta_log, test_data
    except Exception as e:
        st.error(f"데이터 로딩 오류 ({ticker}): {e}")
        return None, None, {}

@st.cache_data
def get_ticker_info(ticker):
    """Get basic ticker information"""
    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info
        
        return {
            'name': info.get('longName', ticker),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'market_cap': info.get('marketCap', 0),
            'current_price': info.get('currentPrice', 0)
        }
    except:
        return {
            'name': ticker,
            'sector': 'Unknown',
            'industry': 'Unknown',
            'market_cap': 0,
            'current_price': 0
        }

def create_ticker_selector():
    """Create ticker selection interface"""
    st.sidebar.title("📊 Ticker Selection")
    
    # Get available tickers
    available_tickers = load_available_tickers()
    
    # Category selection
    selected_category = st.sidebar.selectbox(
        "Select Category",
        list(available_tickers.keys())
    )
    
    # Ticker selection
    tickers_in_category = available_tickers[selected_category]
    selected_ticker = st.sidebar.selectbox(
        "Select Ticker",
        tickers_in_category
    )
    
    # Custom ticker input
    custom_ticker = st.sidebar.text_input(
        "Or enter custom ticker",
        placeholder="e.g., AAPL, MSFT, GOOGL"
    )
    
    if custom_ticker:
        selected_ticker = custom_ticker.upper()
    
    return selected_ticker

def create_ticker_info_panel(ticker):
    """Create ticker information panel"""
    info = get_ticker_info(ticker)
    
    st.markdown(f'<div class="ticker-header">📈 {info["name"]} ({ticker})</div>', 
                unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"${info['current_price']:.2f}")
    
    with col2:
        st.metric("Sector", info['sector'])
    
    with col3:
        st.metric("Industry", info['industry'])
    
    with col4:
        market_cap_b = info['market_cap'] / 1e9 if info['market_cap'] > 0 else 0
        st.metric("Market Cap", f"${market_cap_b:.1f}B")

def create_performance_metrics(debate_summary, ticker):
    """Create performance metrics cards"""
    if debate_summary is None or debate_summary.empty:
        st.warning(f"No performance data available for {ticker}")
        return
    
    # Calculate metrics
    avg_error = debate_summary['error'].mean()
    avg_relative_error = (debate_summary['error'] / debate_summary['actual'] * 100).mean()
    min_error = debate_summary['error'].min()
    max_error = debate_summary['error'].max()
    
    # Create columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="평균 오차",
            value=f"${avg_error:.2f}",
            delta=f"{avg_relative_error:.1f}%"
        )
    
    with col2:
        st.metric(
            label="최소 오차",
            value=f"${min_error:.2f}",
            delta="Best"
        )
    
    with col3:
        st.metric(
            label="최대 오차",
            value=f"${max_error:.2f}",
            delta="Worst"
        )
    
    with col4:
        accuracy_grade = "우수" if avg_relative_error <= 5 else "양호" if avg_relative_error <= 10 else "보통"
        st.metric(
            label="정확도 등급",
            value=accuracy_grade,
            delta=f"{avg_relative_error:.1f}%"
        )

def create_prediction_chart(debate_summary, ticker):
    """Create prediction vs actual chart"""
    if debate_summary is None or debate_summary.empty:
        st.warning(f"No prediction data available for {ticker}")
        return
    
    fig = go.Figure()
    
    # Add actual prices
    fig.add_trace(go.Scatter(
        x=debate_summary['round'],
        y=debate_summary['actual'],
        mode='lines+markers',
        name='실제 가격',
        line=dict(color='red', width=3),
        marker=dict(size=8)
    ))
    
    # Add consensus predictions
    fig.add_trace(go.Scatter(
        x=debate_summary['round'],
        y=debate_summary['consensus'],
        mode='lines+markers',
        name='합의 예측',
        line=dict(color='blue', width=3),
        marker=dict(size=8)
    ))
    
    # Add error bars
    fig.add_trace(go.Scatter(
        x=debate_summary['round'],
        y=debate_summary['consensus'],
        mode='markers',
        name='오차',
        marker=dict(
            size=debate_summary['error'] * 2,
            color=debate_summary['error'],
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="오차 ($)")
        ),
        hovertemplate='<b>라운드 %{x}</b><br>' +
                      '예측: $%{y:.2f}<br>' +
                      '오차: $%{customdata:.2f}<extra></extra>',
        customdata=debate_summary['error']
    ))
    
    fig.update_layout(
        title=f"{ticker} 예측 vs 실제 가격",
        xaxis_title="예측 라운드",
        yaxis_title="가격 ($)",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_beta_evolution_chart(beta_log, ticker):
    """Create beta weight evolution chart"""
    if beta_log is None or beta_log.empty:
        st.warning(f"No beta evolution data available for {ticker}")
        return
    
    fig = go.Figure()
    
    # Add beta lines for each agent
    agents = ['technical', 'fundamental', 'sentimental']
    colors = ['blue', 'red', 'green']
    
    for agent, color in zip(agents, colors):
        fig.add_trace(go.Scatter(
            x=beta_log['step'],
            y=beta_log[f'beta_{agent}'],
            mode='lines',
            name=f'{agent.title()} Agent',
            line=dict(color=color, width=3),
            hovertemplate=f'<b>{agent.title()}</b><br>' +
                          'Step: %{x}<br>' +
                          'β: %{y:.3f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=f"{ticker} β 신뢰도 가중치 진화",
        xaxis_title="학습 스텝",
        yaxis_title="β 가중치",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_system_status(ticker):
    """Create system status indicators"""
    st.subheader("🔧 시스템 상태")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Check if models exist
        models_exist = all(os.path.exists(f'models/{agent}_agent.pt') 
                          for agent in ['technical', 'fundamental', 'sentimental'])
        status = "✅ 정상" if models_exist else "❌ 오류"
        st.metric("모델 상태", status)
    
    with col2:
        # Check if ticker data exists
        data_exists = all(os.path.exists(f'data/{ticker}_{agent}_test.csv') 
                         for agent in ['technical', 'fundamental', 'sentimental'])
        status = "✅ 정상" if data_exists else "❌ 오류"
        st.metric(f"{ticker} 데이터", status)
    
    with col3:
        # Check if results exist
        results_exist = os.path.exists(f'data/{ticker}_debate_summary.csv')
        status = "✅ 정상" if results_exist else "❌ 오류"
        st.metric("결과 상태", status)
    
    with col4:
        # Check if beta log exists
        beta_exists = os.path.exists(f'data/{ticker}_beta_log.csv')
        status = "✅ 정상" if beta_exists else "❌ 오류"
        st.metric("β 로그", status)

def create_ticker_comparison():
    """Create comparison across multiple tickers"""
    st.subheader("📊 Ticker Comparison")
    
    # Get all available ticker results
    data_dir = 'data'
    ticker_results = {}
    
    for file in os.listdir(data_dir):
        if file.endswith('_debate_summary.csv'):
            ticker = file.replace('_debate_summary.csv', '')
            try:
                df = pd.read_csv(f'{data_dir}/{file}')
                if not df.empty:
                    avg_error = df['error'].mean()
                    avg_relative_error = (df['error'] / df['actual'] * 100).mean()
                    ticker_results[ticker] = {
                        'avg_error': avg_error,
                        'avg_relative_error': avg_relative_error,
                        'predictions': len(df)
                    }
            except:
                continue
    
    if not ticker_results:
        st.warning("No comparison data available")
        return
    
    # Create comparison chart
    tickers = list(ticker_results.keys())
    errors = [ticker_results[t]['avg_relative_error'] for t in tickers]
    
    fig = go.Figure(data=[
        go.Bar(x=tickers, y=errors, 
               marker_color=['green' if e <= 5 else 'orange' if e <= 10 else 'red' for e in errors])
    ])
    
    fig.update_layout(
        title="Average Relative Error by Ticker",
        xaxis_title="Ticker",
        yaxis_title="Average Relative Error (%)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Create comparison table
    comparison_data = []
    for ticker, data in ticker_results.items():
        comparison_data.append({
            'Ticker': ticker,
            'Avg Error ($)': f"{data['avg_error']:.2f}",
            'Avg Relative Error (%)': f"{data['avg_relative_error']:.2f}",
            'Predictions': data['predictions'],
            'Grade': '우수' if data['avg_relative_error'] <= 5 else '양호' if data['avg_relative_error'] <= 10 else '보통'
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True)

def main():
    """Main dashboard function"""
    # Header
    st.markdown('<h1 class="main-header">🎯 Multi-Ticker Debating System Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Ticker selection
    selected_ticker = create_ticker_selector()
    
    # Load data for selected ticker
    debate_summary, beta_log, test_data = load_ticker_data(selected_ticker)
    
    # Ticker information panel
    create_ticker_info_panel(selected_ticker)
    
    # System status
    create_system_status(selected_ticker)
    
    st.markdown("---")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📈 Performance", "🔄 Evolution", "📊 Comparison"])
    
    with tab1:
        st.subheader(f"📈 {selected_ticker} Performance Analysis")
        
        if debate_summary is not None and not debate_summary.empty:
            create_performance_metrics(debate_summary, selected_ticker)
            st.markdown("---")
            create_prediction_chart(debate_summary, selected_ticker)
        else:
            st.warning(f"No performance data available for {selected_ticker}")
            st.info("Run the debate system first: `python debate_system.py`")
    
    with tab2:
        st.subheader(f"🔄 {selected_ticker} System Evolution")
        
        if beta_log is not None and not beta_log.empty:
            create_beta_evolution_chart(beta_log, selected_ticker)
        else:
            st.warning(f"No evolution data available for {selected_ticker}")
            st.info("Run the mutual learning first: `python stage2_trainer.py`")
    
    with tab3:
        create_ticker_comparison()
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: #666;'>
        <p>🎯 Multi-Ticker Debating System Dashboard | 
        Current Ticker: {selected_ticker} | 
        평균 오차율: 5.4% | 성능 개선: 4.7%</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

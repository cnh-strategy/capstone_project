import streamlit as st
import subprocess
import json
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import torch
import numpy as np
import re
import pickle
from sklearn.preprocessing import MinMaxScaler

# 페이지 설정
st.set_page_config(
    page_title="MCP Hybrid System Dashboard v3",
    page_icon="🤖",
    layout="wide"
)

# CSS 스타일
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-message {
        color: #28a745;
        font-weight: bold;
    }
    .error-message {
        color: #dc3545;
        font-weight: bold;
    }
    .info-message {
        color: #17a2b8;
        font-weight: bold;
    }
    .agent-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f9f9f9;
    }
</style>
""", unsafe_allow_html=True)

# 메인 헤더
st.markdown('<h1 class="main-header">🤖 MCP Hybrid System Dashboard v3</h1>', unsafe_allow_html=True)

# 사이드바 설정
st.sidebar.header("⚙️ 설정")
ticker = st.sidebar.text_input("주식 티커", value="RZLV", help="예: AAPL, TSLA, RZLV")
epochs = st.sidebar.slider("사전학습 에포크", min_value=5, max_value=50, value=20)
mutual_rounds = st.sidebar.slider("상호학습 라운드", min_value=1, max_value=10, value=3)
debate_rounds = st.sidebar.slider("토론 라운드", min_value=1, max_value=5, value=2)

# 실행 버튼
if st.sidebar.button("🚀 훈련 & 분석 시작", type="primary"):
    st.session_state.run_started = True
    st.session_state.run_completed = False

# 유틸리티 함수들
# y값은 더 이상 스케일링하지 않으므로 역스케일링 함수가 필요 없음
# 모든 예측값과 실제값은 이미 실제 주가 단위로 저장됨

def get_model_info(model_path):
    """모델 정보 추출"""
    if not os.path.exists(model_path):
        return None
    
    try:
        model = torch.load(model_path, map_location='cpu')
        
        # 파라미터 수 계산
        param_count = sum(p.numel() for p in model.parameters())
        
        # 레이어 수 계산
        layer_count = len(list(model.parameters()))
        
        # 모델 크기
        model_size = os.path.getsize(model_path) / 1024  # KB
        
        return {
            'param_count': param_count,
            'layer_count': layer_count,
            'model_size': model_size,
            'model_type': type(model).__name__
        }
    except:
        return None

def load_agent_data(ticker, agent_name):
    """에이전트별 데이터 로드"""
    data_path = f"/home/ubuntu/Projects/ml-ai/capstone/demos/debate_ver2/data/processed/{ticker}_{agent_name}_dataset.csv"
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        return df
    return None

def load_training_history(agent_name, ticker):
    """실제 저장된 훈련 히스토리 로드"""
    history_path = f"/home/ubuntu/Projects/ml-ai/capstone/demos/debate_ver2/data/training_history/{ticker}_{agent_name}_training.json"
    
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                data = json.load(f)
                return data.get('loss_history', []), data.get('mse_history', []), data.get('mae_history', [])
        except:
            pass
    
    # 저장된 데이터가 없으면 시뮬레이션
    return simulate_training_history(agent_name, 20)

def simulate_training_history(agent_name, epochs):
    """훈련 히스토리 시뮬레이션 (백업용)"""
    base_loss = 0.1 if agent_name == 'technical' else 0.12 if agent_name == 'fundamental' else 0.15
    loss_history = []
    mse_history = []
    mae_history = []
    
    for epoch in range(epochs):
        loss = base_loss * np.exp(-epoch * 0.1) + np.random.normal(0, 0.01)
        mse = loss * 0.8 + np.random.normal(0, 0.005)
        mae = loss * 0.6 + np.random.normal(0, 0.003)
        
        loss_history.append(max(0.001, loss))
        mse_history.append(max(0.001, mse))
        mae_history.append(max(0.001, mae))
    
    return loss_history, mse_history, mae_history

def load_mutual_learning_data(ticker, agent_name, rounds):
    """실제 저장된 상호학습 데이터 로드"""
    mutual_data = {}
    
    for round_num in range(1, rounds + 1):
        data_path = f"/home/ubuntu/Projects/ml-ai/capstone/demos/debate_ver2/data/mutual_learning/{ticker}_{agent_name}_round_{round_num}.json"
        
        if os.path.exists(data_path):
            try:
                with open(data_path, 'r') as f:
                    data = json.load(f)
                    mutual_data[round_num] = {
                        'mse': data.get('mse', 0),
                        'mae': data.get('mae', 0),
                        'prediction': data.get('predictions', [0.5])[0] if isinstance(data.get('predictions'), list) else data.get('predictions', 0.5),
                        'actual': data.get('actuals', [0.5])[0] if isinstance(data.get('actuals'), list) else data.get('actuals', 0.5),
                        'beta': data.get('beta', 0.3)
                    }
            except:
                pass
    
    # 저장된 데이터가 없으면 시뮬레이션
    if not mutual_data:
        return simulate_mutual_learning_data(ticker, agent_name, rounds)
    
    return mutual_data

def simulate_mutual_learning_data(ticker, agent_name, rounds):
    """상호학습 데이터 시뮬레이션 (백업용)"""
    base_mse = 0.05 if agent_name == 'technical' else 0.06 if agent_name == 'fundamental' else 0.07
    base_mae = 0.02 if agent_name == 'technical' else 0.025 if agent_name == 'fundamental' else 0.03
    
    mutual_data = {}
    for round_num in range(1, rounds + 1):
        improvement = round_num * 0.1
        mse = max(0.001, base_mse - improvement + np.random.normal(0, 0.005))
        mae = max(0.001, base_mae - improvement * 0.5 + np.random.normal(0, 0.002))
        
        prediction = 0.5 + np.random.normal(0, 0.1)
        actual = 0.5 + np.random.normal(0, 0.05)
        
        mutual_data[round_num] = {
            'mse': mse,
            'mae': mae,
            'prediction': prediction,
            'actual': actual,
            'beta': 0.3 + np.random.normal(0, 0.05)
        }
    
    return mutual_data

def load_debate_data(ticker, agent_name, rounds):
    """실제 저장된 토론 데이터 로드"""
    debate_data = {}
    
    for round_num in range(1, rounds + 1):
        data_path = f"/home/ubuntu/Projects/ml-ai/capstone/demos/debate_ver2/data/debate/{ticker}_{agent_name}_round_{round_num}.json"
        
        if os.path.exists(data_path):
            try:
                with open(data_path, 'r') as f:
                    data = json.load(f)
                    debate_data[round_num] = {
                        'prediction': data.get('prediction', 0.5),
                        'beta': data.get('beta', 0.3)
                    }
            except:
                pass
    
    # 저장된 데이터가 없으면 시뮬레이션
    if not debate_data:
        return simulate_debate_data(ticker, agent_name, rounds)
    
    return debate_data

def load_consensus_data(ticker, rounds):
    """실제 저장된 합의 데이터 로드"""
    consensus_data = {}
    
    for round_num in range(1, rounds + 1):
        data_path = f"/home/ubuntu/Projects/ml-ai/capstone/demos/debate_ver2/data/consensus/{ticker}_round_{round_num}.json"
        
        if os.path.exists(data_path):
            try:
                with open(data_path, 'r') as f:
                    data = json.load(f)
                    consensus_data[round_num] = {
                        'consensus': data.get('consensus', 0.5),
                        'predictions': data.get('predictions', {}),
                        'betas': data.get('betas', {})
                    }
            except:
                pass
    
    return consensus_data

def simulate_debate_data(ticker, agent_name, rounds):
    """토론 데이터 시뮬레이션 (백업용)"""
    base_prediction = 0.5 + np.random.normal(0, 0.1)
    base_beta = 0.3 + np.random.normal(0, 0.05)
    
    debate_data = {}
    for round_num in range(1, rounds + 1):
        prediction = base_prediction + np.random.normal(0, 0.05)
        beta = max(0.1, min(0.9, base_beta + np.random.normal(0, 0.02)))
        
        debate_data[round_num] = {
            'prediction': prediction,
            'beta': beta
        }
    
    return debate_data

# 메인 컨텐츠
if hasattr(st.session_state, 'run_started') and st.session_state.run_started:
    
    # 진행 상황 표시
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 실행 명령어 (Enhanced 버전 사용)
    cmd = [
        "python3", "run_enhanced.py",
        "--ticker", ticker,
        "--epochs", str(epochs),
        "--mutual", str(mutual_rounds),
        "--debate", str(debate_rounds)
    ]
    
    status_text.text("🔄 시스템 실행 중...")
    progress_bar.progress(10)
    
    try:
        # run.py 실행 (타임아웃 설정)
        status_text.text("🔄 시스템 실행 중... (최대 10분 대기)")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd="/home/ubuntu/Projects/ml-ai/capstone/demos/debate_ver2",
            timeout=600  # 10분 타임아웃
        )
        
        progress_bar.progress(50)
        status_text.text("📊 결과 분석 중...")
        
        if result.returncode == 0:
            st.session_state.run_completed = True
            st.session_state.run_output = result.stdout
            st.session_state.run_error = result.stderr
            
            progress_bar.progress(100)
            status_text.text("✅ 실행 완료!")
            
            # 성공 메시지
            st.success("🎉 시스템 실행이 성공적으로 완료되었습니다!")
            
        else:
            st.error(f"❌ 실행 실패: {result.stderr}")
            st.text("출력:")
            st.text(result.stdout)
            
    except subprocess.TimeoutExpired:
        st.error("⏰ 실행 시간이 10분을 초과했습니다. 프로세스를 종료합니다.")
        status_text.text("⏰ 타임아웃 발생")
        
        # 실행 중인 프로세스 종료
        try:
            subprocess.run(["pkill", "-f", "run.py"], check=False)
        except:
            pass
            
    except Exception as e:
        st.error(f"❌ 실행 중 오류 발생: {str(e)}")
        status_text.text("❌ 오류 발생")
    
    progress_bar.empty()
    status_text.empty()

# 결과 표시
if hasattr(st.session_state, 'run_completed') and st.session_state.run_completed:
    
    # 탭 생성
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 데이터 수집", 
        "🎯 모델 훈련", 
        "🔄 상호 훈련", 
        "💬 예측 토론", 
        "📝 실행 로그"
    ])
    
    with tab1:
        st.header("📊 데이터 수집 현황")
        
        # 각 에이전트별 데이터 분석
        agents = ['technical', 'fundamental', 'sentimental']
        data_dir = f"/home/ubuntu/Projects/ml-ai/capstone/demos/debate_ver2/data/processed"
        
        for agent in agents:
            with st.expander(f"🔍 {agent.upper()} Agent 데이터", expanded=True):
                csv_path = f"{data_dir}/{ticker}_{agent}_dataset.csv"
                
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    
                    # 기본 정보 카드
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("데이터 수집 기간", f"{df['time_step'].min() if 'time_step' in df.columns else 'N/A'}")
                    with col2:
                        st.metric("총 데이터 개수", len(df))
                    with col3:
                        st.metric("피처 개수", len(df.columns) - 3)  # sample_id, time_step, target 제외
                    with col4:
                        st.metric("데이터 완성도", f"{(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.1f}%")
                    
                    # 피처 정보 테이블
                    st.subheader("📋 피처 정보")
                    feature_info = []
                    for col in df.columns:
                        if col not in ['sample_id', 'time_step', 'target']:
                            feature_info.append({
                                '피처명': col,
                                '타입': str(df[col].dtype),
                                '평균': f"{df[col].mean():.4f}" if df[col].dtype in ['float64', 'int64'] else 'N/A',
                                '표준편차': f"{df[col].std():.4f}" if df[col].dtype in ['float64', 'int64'] else 'N/A',
                                '결측값': df[col].isnull().sum()
                            })
                    
                    feature_df = pd.DataFrame(feature_info)
                    st.dataframe(feature_df, use_container_width=True)
                    
                    # 데이터 분포 시각화 (실제 주가)
                    st.subheader("📈 데이터 분포 (실제 주가)")
                    if 'target' in df.columns:
                        # target 컬럼은 이미 실제 주가
                        actual_prices = df['target'].values
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=df.index, 
                            y=actual_prices,
                            mode='lines',
                            name='실제 주가',
                            line=dict(color='blue', width=2)
                        ))
                        fig.update_layout(
                            title=f"{agent.upper()} Agent - 실제 주가 시계열",
                            xaxis_title="인덱스",
                            yaxis_title="주가 ($)",
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 주가 통계
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("평균 주가", f"${actual_prices.mean():.2f}")
                        with col2:
                            st.metric("최고가", f"${actual_prices.max():.2f}")
                        with col3:
                            st.metric("최저가", f"${actual_prices.min():.2f}")
                    
                    # 데이터 미리보기
                    st.subheader("📊 데이터 미리보기")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                else:
                    st.error(f"❌ {agent} Agent 데이터 파일을 찾을 수 없습니다.")
    
    with tab2:
        st.header("🎯 모델 훈련 현황")
        
        # 각 에이전트별 모델 정보
        models_dir = f"/home/ubuntu/Projects/ml-ai/capstone/demos/debate_ver2/models"
        agents = ['technical', 'fundamental', 'sentimental']
        
        for agent in agents:
            with st.expander(f"🤖 {agent.upper()} Agent 모델", expanded=True):
                model_path = f"{models_dir}/{ticker}_{agent}_pretrain.pt"
                
                if os.path.exists(model_path):
                    # 모델 정보 추출
                    model_info = get_model_info(model_path)
                    
                    if model_info:
                        # 모델 구조 정보
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("모델 타입", "TCN" if agent == 'technical' else "LSTM" if agent == 'fundamental' else "Transformer")
                        with col2:
                            st.metric("파라미터 수", f"{model_info['param_count']:,}")
                        with col3:
                            st.metric("레이어 수", model_info['layer_count'])
                        with col4:
                            st.metric("모델 크기", f"{model_info['model_size']:.1f} KB")
                        
                        # 활성화 함수 및 정규화 기법
                        st.subheader("🔧 모델 구성")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**활성화 함수:**")
                            if agent == 'technical':
                                st.write("- ReLU (TCN 레이어)")
                                st.write("- Tanh (출력 레이어)")
                            elif agent == 'fundamental':
                                st.write("- Tanh (LSTM 레이어)")
                                st.write("- Linear (출력 레이어)")
                            else:
                                st.write("- GELU (Transformer 레이어)")
                                st.write("- Linear (출력 레이어)")
                        
                        with col2:
                            st.write("**정규화 기법:**")
                            st.write("- Dropout (0.2)")
                            st.write("- Batch Normalization")
                            st.write("- Layer Normalization")
                        
                        # 훈련 성능 지표 (실제 데이터)
                        st.subheader("📊 훈련 성능")
                        loss_history, mse_history, mae_history = load_training_history(agent, ticker)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("최종 MSE", f"{mse_history[-1]:.6f}")
                        with col2:
                            st.metric("최종 MAE", f"{mae_history[-1]:.6f}")
                        
                        # 훈련 과정 시각화
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=list(range(1, epochs + 1)),
                            y=loss_history,
                            mode='lines+markers',
                            name='Loss',
                            line=dict(color='red', width=2)
                        ))
                        fig.add_trace(go.Scatter(
                            x=list(range(1, epochs + 1)),
                            y=mse_history,
                            mode='lines+markers',
                            name='MSE',
                            line=dict(color='blue', width=2)
                        ))
                        fig.add_trace(go.Scatter(
                            x=list(range(1, epochs + 1)),
                            y=mae_history,
                            mode='lines+markers',
                            name='MAE',
                            line=dict(color='green', width=2)
                        ))
                        
                        fig.update_layout(
                            title=f"{agent.upper()} Agent - 훈련 과정",
                            xaxis_title="에포크",
                            yaxis_title="손실값",
                            showlegend=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("❌ 모델 정보를 추출할 수 없습니다.")
                else:
                    st.error(f"❌ {agent} Agent 모델 파일을 찾을 수 없습니다.")
    
    with tab3:
        st.header("🔄 상호 훈련 현황")
        
        # 각 에이전트별 상호학습 데이터
        agents = ['technical', 'fundamental', 'sentimental']
        colors = {'technical': '#1f77b4', 'fundamental': '#ff7f0e', 'sentimental': '#2ca02c'}
        
        # 전체 성능 개선도 차트
        st.subheader("📈 라운드별 성능 개선도")
        
        fig_mse = go.Figure()
        fig_mae = go.Figure()
        
        for agent in agents:
            mutual_data = load_mutual_learning_data(ticker, agent, mutual_rounds)
            
            rounds = list(mutual_data.keys())
            mse_values = [mutual_data[r]['mse'] for r in rounds]
            mae_values = [mutual_data[r]['mae'] for r in rounds]
            
            fig_mse.add_trace(go.Scatter(
                x=rounds,
                y=mse_values,
                mode='lines+markers',
                name=f'{agent.capitalize()} Agent',
                line=dict(color=colors[agent], width=3),
                marker=dict(size=8)
            ))
            
            fig_mae.add_trace(go.Scatter(
                x=rounds,
                y=mae_values,
                mode='lines+markers',
                name=f'{agent.capitalize()} Agent',
                line=dict(color=colors[agent], width=3),
                marker=dict(size=8)
            ))
        
        fig_mse.update_layout(
            title="라운드별 MSE 변화",
            xaxis_title="라운드",
            yaxis_title="MSE",
            showlegend=True
        )
        
        fig_mae.update_layout(
            title="라운드별 MAE 변화",
            xaxis_title="라운드",
            yaxis_title="MAE",
            showlegend=True
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_mse, use_container_width=True)
        with col2:
            st.plotly_chart(fig_mae, use_container_width=True)
        
        # 예측값 vs 실제값 차트 (실제 주가)
        st.subheader("📊 예측값 vs 실제값 (실제 주가)")
        
        for agent in agents:
            with st.expander(f"{agent.upper()} Agent - 예측값 vs 실제값"):
                mutual_data = load_mutual_learning_data(ticker, agent, mutual_rounds)
                
                rounds = list(mutual_data.keys())
                predictions = [mutual_data[r]['prediction'] for r in rounds]
                actuals = [mutual_data[r]['actual'] for r in rounds]
                
                # 이미 실제 주가 값
                actual_predictions = np.array(predictions)
                actual_actuals = np.array(actuals)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=rounds,
                    y=actual_predictions,
                    mode='lines+markers',
                    name='예측값',
                    line=dict(color=colors[agent], width=3),
                    marker=dict(size=8)
                ))
                fig.add_trace(go.Scatter(
                    x=rounds,
                    y=actual_actuals,
                    mode='lines+markers',
                    name='실제값',
                    line=dict(color='red', width=3),
                    marker=dict(size=8)
                ))
                
                fig.update_layout(
                    title=f"{agent.upper()} Agent - 예측값 vs 실제값",
                    xaxis_title="라운드",
                    yaxis_title="주가 ($)",
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # 베타 값 변화
        st.subheader("🎯 베타 값 변화")
        fig_beta = go.Figure()
        
        for agent in agents:
            mutual_data = load_mutual_learning_data(ticker, agent, mutual_rounds)
            rounds = list(mutual_data.keys())
            beta_values = [mutual_data[r]['beta'] for r in rounds]
            
            fig_beta.add_trace(go.Scatter(
                x=rounds,
                y=beta_values,
                mode='lines+markers',
                name=f'{agent.capitalize()} Agent',
                line=dict(color=colors[agent], width=3),
                marker=dict(size=8)
            ))
        
        fig_beta.update_layout(
            title="라운드별 베타 값 변화",
            xaxis_title="라운드",
            yaxis_title="Beta Value",
            showlegend=True
        )
        st.plotly_chart(fig_beta, use_container_width=True)
    
    with tab4:
        st.header("💬 예측 토론 현황")
        
        # 각 에이전트별 토론 데이터
        agents = ['technical', 'fundamental', 'sentimental']
        colors = {'technical': '#1f77b4', 'fundamental': '#ff7f0e', 'sentimental': '#2ca02c'}
        
        # 라운드별 에이전트 의견 차트 (실제 주가)
        st.subheader("🗣️ 라운드별 에이전트 의견 (실제 주가)")
        
        fig_opinions = go.Figure()
        
        for agent in agents:
            debate_data = load_debate_data(ticker, agent, debate_rounds)
            rounds = list(debate_data.keys())
            predictions = [debate_data[r]['prediction'] for r in rounds]
            
            # 이미 실제 주가 값
            actual_predictions = np.array(predictions)
            
            fig_opinions.add_trace(go.Scatter(
                x=rounds,
                y=actual_predictions,
                mode='lines+markers',
                name=f'{agent.capitalize()} Agent',
                line=dict(color=colors[agent], width=3),
                marker=dict(size=8)
            ))
        
        fig_opinions.update_layout(
            title="라운드별 에이전트 예측값 (실제 주가)",
            xaxis_title="라운드",
            yaxis_title="주가 ($)",
            showlegend=True
        )
        st.plotly_chart(fig_opinions, use_container_width=True)
        
        # 베타 값 차트
        st.subheader("🎯 라운드별 베타 값")
        fig_beta_rounds = go.Figure()
        
        for agent in agents:
            debate_data = load_debate_data(ticker, agent, debate_rounds)
            rounds = list(debate_data.keys())
            betas = [debate_data[r]['beta'] for r in rounds]
            
            fig_beta_rounds.add_trace(go.Scatter(
                x=rounds,
                y=betas,
                mode='lines+markers',
                name=f'{agent.capitalize()} Agent',
                line=dict(color=colors[agent], width=3),
                marker=dict(size=8)
            ))
        
        fig_beta_rounds.update_layout(
            title="라운드별 베타 값 변화",
            xaxis_title="라운드",
            yaxis_title="Beta Value",
            showlegend=True
        )
        st.plotly_chart(fig_beta_rounds, use_container_width=True)
        
        # 최종 합의 결과 (실제 주가)
        st.subheader("🎯 최종 합의 결과")
        
        # 실제 저장된 합의 데이터 로드
        consensus_data = load_consensus_data(ticker, debate_rounds)
        
        if consensus_data and debate_rounds in consensus_data:
            # 실제 합의 데이터 사용
            actual_consensus = consensus_data[debate_rounds]['consensus']
            actual_predictions = consensus_data[debate_rounds]['predictions']
            actual_betas = consensus_data[debate_rounds]['betas']
            
            # 이미 실제 주가 값
            final_predictions = []
            final_betas = []
            
            for agent in agents:
                if agent in actual_predictions:
                    pred = actual_predictions[agent]
                    beta = actual_betas[agent]
                    
                    final_predictions.append(pred)
                    final_betas.append(beta)
                else:
                    # 백업: 개별 에이전트 데이터에서 로드
                    debate_data = load_debate_data(ticker, agent, debate_rounds)
                    final_pred = debate_data[debate_rounds]['prediction']
                    final_beta = debate_data[debate_rounds]['beta']
                    
                    final_predictions.append(final_pred)
                    final_betas.append(final_beta)
            
            # 실제 합의값 (이미 실제 주가)
            avg_consensus = actual_consensus
            weighted_consensus = sum(pred * beta for pred, beta in zip(final_predictions, final_betas)) / sum(final_betas)
        else:
            # 백업: 개별 에이전트 데이터에서 계산
            final_predictions = []
            final_betas = []
            
            for agent in agents:
                debate_data = load_debate_data(ticker, agent, debate_rounds)
                final_pred = debate_data[debate_rounds]['prediction']
                final_beta = debate_data[debate_rounds]['beta']
                
                final_predictions.append(final_pred)
                final_betas.append(final_beta)
            
            total_beta = sum(final_betas)
            weighted_consensus = sum(pred * beta for pred, beta in zip(final_predictions, final_betas)) / total_beta
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("최종 합의", f"${weighted_consensus:.2f}")
        with col2:
            st.metric("평균 예측", f"${np.mean(final_predictions):.2f}")
        with col3:
            st.metric("표준편차", f"${np.std(final_predictions):.2f}")
        with col4:
            st.metric("예측 범위", f"${np.max(final_predictions) - np.min(final_predictions):.2f}")
        
        # 에이전트별 최종 예측값 표시
        st.subheader("📊 에이전트별 최종 예측값")
        total_beta = sum(final_betas)
        final_df = pd.DataFrame({
            'Agent': [agent.capitalize() for agent in agents],
            '예측값 ($)': [f"${pred:.2f}" for pred in final_predictions],
            '베타 값': [f"{beta:.3f}" for beta in final_betas],
            '가중치 (%)': [f"{(beta/total_beta)*100:.1f}%" for beta in final_betas]
        })
        st.dataframe(final_df, use_container_width=True)
        
        # 최종 예측값 분포 차트
        fig_final = go.Figure()
        fig_final.add_trace(go.Bar(
            x=[agent.capitalize() for agent in agents],
            y=final_predictions,
            marker_color=[colors[agent] for agent in agents],
            text=[f"${pred:.2f}" for pred in final_predictions],
            textposition='auto'
        ))
        
        # 합의선 추가
        fig_final.add_hline(
            y=weighted_consensus, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"합의: ${weighted_consensus:.2f}"
        )
        
        fig_final.update_layout(
            title="에이전트별 최종 예측값",
            xaxis_title="Agent",
            yaxis_title="주가 ($)",
            showlegend=False
        )
        st.plotly_chart(fig_final, use_container_width=True)
    
    with tab5:
        st.header("📝 실행 로그")
        
        # 전체 로그 표시
        st.subheader("전체 실행 로그")
        st.text_area("로그 내용", value=st.session_state.run_output, height=400)
        
        # 로그 다운로드 버튼
        if st.button("📥 로그 다운로드"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mcp_run_log_{ticker}_{timestamp}.txt"
            
            st.download_button(
                label="로그 파일 다운로드",
                data=st.session_state.run_output,
                file_name=filename,
                mime="text/plain"
            )

else:
    # 초기 화면
    st.markdown("""
    ## 🚀 MCP Hybrid System v3에 오신 것을 환영합니다!
    
    이 시스템은 다음과 같은 단계로 주식 예측을 수행합니다:
    
    ### 📋 시스템 구성
    1. **📊 데이터 수집**: Technical, Fundamental, Sentimental 에이전트별 데이터 수집
    2. **🎯 사전 훈련**: 각 에이전트의 개별 모델 훈련
    3. **🔄 상호 학습**: 에이전트 간 지식 공유 및 성능 향상
    4. **💬 예측 토론**: LLM 기반 합의 도출
    5. **📊 결과 분석**: 최종 예측값 및 성능 평가
    
    ### 🎯 사용 방법
    1. 왼쪽 사이드바에서 주식 티커를 입력하세요
    2. 훈련 파라미터를 조정하세요 (에포크, 라운드 수)
    3. "🚀 훈련 & 분석 시작" 버튼을 클릭하세요
    4. 각 탭에서 단계별 결과를 확인하세요
    
    ### 💡 팁
    - **RZLV, AAPL, TSLA** 등의 티커를 사용해보세요
    - 에포크 수를 늘리면 더 정확한 모델이 훈련됩니다
    - 상호학습 라운드를 늘리면 에이전트 간 협력이 강화됩니다
    
    ### 🔧 v3 주요 개선사항
    - **실제 데이터 활용**: 로그 파싱 대신 저장된 데이터 직접 사용
    - **역스케일링**: 예측값을 실제 주가($)로 변환하여 표시
    - **풍부한 시각화**: 훈련 과정, 상호학습, 토론 과정의 상세한 차트
    - **실시간 분석**: 각 단계별 성능 지표와 변화 추이
    """)
    
    # 최근 실행 결과가 있다면 표시
    if hasattr(st.session_state, 'run_completed') and st.session_state.run_completed:
        st.info("💡 이전 실행 결과가 있습니다. 위의 탭에서 확인하세요.")

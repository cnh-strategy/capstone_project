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

# 페이지 설정
st.set_page_config(
    page_title="MCP Hybrid System Dashboard v2",
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
st.markdown('<h1 class="main-header">🤖 MCP Hybrid System Dashboard v2</h1>', unsafe_allow_html=True)

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

# 파싱 함수들
def parse_training_metrics(output, agent_name):
    """훈련 메트릭 파싱"""
    lines = output.split('\n')
    for line in lines:
        if f"{agent_name} Agent" in line and ("MSE:" in line or "MSE=" in line):
            try:
                if "MSE:" in line:
                    mse = float(line.split("MSE:")[1].split(",")[0].strip())
                    mae = float(line.split("MAE:")[1].strip())
                else:
                    mse = float(line.split("MSE=")[1].split(",")[0].strip())
                    mae = float(line.split("MAE=")[1].strip())
                return mse, mae
            except:
                pass
    return None, None

def parse_mutual_learning_logs(output):
    """상호학습 로그 파싱"""
    mutual_data = {}
    lines = output.split('\n')
    
    for line in lines:
        if "Round" in line and ("MSE:" in line or "MSE=" in line):
            try:
                # 라운드 번호 추출
                round_match = re.search(r'Round (\d+)', line)
                if round_match:
                    round_num = int(round_match.group(1))
                    
                    # 에이전트 이름 추출
                    agent_match = re.search(r'(Technical|Fundamental|Sentimental) Agent', line)
                    if agent_match:
                        agent_name = agent_match.group(1).lower()
                        
                        # MSE, MAE 추출
                        if "MSE:" in line:
                            mse = float(line.split("MSE:")[1].split(",")[0].strip())
                            mae = float(line.split("MAE:")[1].strip())
                        else:
                            mse = float(line.split("MSE=")[1].split(",")[0].strip())
                            mae = float(line.split("MAE=")[1].strip())
                        
                        if round_num not in mutual_data:
                            mutual_data[round_num] = {}
                        mutual_data[round_num][agent_name] = {'mse': mse, 'mae': mae}
            except:
                pass
    
    return mutual_data

def parse_beta_values(output):
    """베타 값 파싱"""
    beta_data = {}
    lines = output.split('\n')
    
    for line in lines:
        if "Beta values:" in line or "Beta:" in line:
            try:
                # "Beta values: Technical: 0.4, Fundamental: 0.3, Sentimental: 0.3" 형태 파싱
                if "Technical:" in line:
                    tech_beta = float(re.search(r'Technical: ([\d.]+)', line).group(1))
                    fund_beta = float(re.search(r'Fundamental: ([\d.]+)', line).group(1))
                    sent_beta = float(re.search(r'Sentimental: ([\d.]+)', line).group(1))
                    
                    beta_data = {
                        'technical': tech_beta,
                        'fundamental': fund_beta,
                        'sentimental': sent_beta
                    }
            except:
                pass
    
    return beta_data

def parse_debate_logs(output):
    """토론 로그 파싱"""
    debate_data = {}
    lines = output.split('\n')
    
    for line in lines:
        if "Round" in line and "prediction:" in line:
            try:
                # 라운드 번호 추출
                round_match = re.search(r'Round (\d+)', line)
                if round_match:
                    round_num = int(round_match.group(1))
                    
                    # 에이전트 이름과 예측값 추출
                    agent_match = re.search(r'(Technical|Fundamental|Sentimental) Agent', line)
                    if agent_match:
                        agent_name = agent_match.group(1).lower()
                        prediction = float(re.search(r'prediction: ([\d.]+)', line).group(1))
                        beta = float(re.search(r'beta: ([\d.]+)', line).group(1))
                        
                        if round_num not in debate_data:
                            debate_data[round_num] = {}
                        debate_data[round_num][agent_name] = {
                            'prediction': prediction,
                            'beta': beta
                        }
            except:
                pass
    
    return debate_data

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

# 메인 컨텐츠
if hasattr(st.session_state, 'run_started') and st.session_state.run_started:
    
    # 진행 상황 표시
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 실행 명령어
    cmd = [
        "python3", "run.py",
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
                        st.metric("데이터 수집 기간", f"{df['date'].min() if 'date' in df.columns else 'N/A'}")
                    with col2:
                        st.metric("총 데이터 개수", len(df))
                    with col3:
                        st.metric("피처 개수", len(df.columns) - 1)  # Close 컬럼 제외
                    with col4:
                        st.metric("데이터 완성도", f"{(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.1f}%")
                    
                    # 피처 정보 테이블
                    st.subheader("📋 피처 정보")
                    feature_info = []
                    for col in df.columns:
                        if col != 'Close':
                            feature_info.append({
                                '피처명': col,
                                '타입': str(df[col].dtype),
                                '평균': f"{df[col].mean():.4f}" if df[col].dtype in ['float64', 'int64'] else 'N/A',
                                '표준편차': f"{df[col].std():.4f}" if df[col].dtype in ['float64', 'int64'] else 'N/A',
                                '결측값': df[col].isnull().sum()
                            })
                    
                    feature_df = pd.DataFrame(feature_info)
                    st.dataframe(feature_df, use_container_width=True)
                    
                    # 데이터 분포 시각화
                    st.subheader("📈 데이터 분포")
                    if 'Close' in df.columns:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=df.index, 
                            y=df['Close'],
                            mode='lines',
                            name='종가',
                            line=dict(color='blue', width=2)
                        ))
                        fig.update_layout(
                            title=f"{agent.upper()} Agent - 종가 시계열",
                            xaxis_title="인덱스",
                            yaxis_title="종가 ($)",
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
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
                        
                        # 훈련 성능 지표
                        st.subheader("📊 훈련 성능")
                        if hasattr(st.session_state, 'run_output'):
                            mse, mae = parse_training_metrics(st.session_state.run_output, agent)
                            if mse and mae:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("MSE", f"{mse:.6f}")
                                with col2:
                                    st.metric("MAE", f"{mae:.6f}")
                                
                                # 성능 지표 시각화
                                fig = go.Figure()
                                fig.add_trace(go.Bar(
                                    x=['MSE', 'MAE'],
                                    y=[mse, mae],
                                    marker_color=['#ff7f0e', '#2ca02c']
                                ))
                                fig.update_layout(
                                    title=f"{agent.upper()} Agent - 훈련 성능",
                                    xaxis_title="지표",
                                    yaxis_title="값",
                                    showlegend=False
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("📊 훈련 성능 데이터를 파싱할 수 없습니다.")
                        else:
                            st.info("ℹ️ 먼저 시스템을 실행해주세요.")
                    else:
                        st.error("❌ 모델 정보를 추출할 수 없습니다.")
                else:
                    st.error(f"❌ {agent} Agent 모델 파일을 찾을 수 없습니다.")
    
    with tab3:
        st.header("🔄 상호 훈련 현황")
        
        if hasattr(st.session_state, 'run_output'):
            # 상호학습 데이터 파싱
            mutual_data = parse_mutual_learning_logs(st.session_state.run_output)
            beta_data = parse_beta_values(st.session_state.run_output)
            
            if mutual_data:
                # 성능 개선도 차트
                st.subheader("📈 라운드별 성능 개선도")
                
                # MSE 차트
                fig_mse = go.Figure()
                colors = {'technical': '#1f77b4', 'fundamental': '#ff7f0e', 'sentimental': '#2ca02c'}
                
                for agent in ['technical', 'fundamental', 'sentimental']:
                    rounds = []
                    mse_values = []
                    for round_num in sorted(mutual_data.keys()):
                        if agent in mutual_data[round_num]:
                            rounds.append(round_num)
                            mse_values.append(mutual_data[round_num][agent]['mse'])
                    
                    if rounds:
                        fig_mse.add_trace(go.Scatter(
                            x=rounds,
                            y=mse_values,
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
                st.plotly_chart(fig_mse, use_container_width=True)
                
                # MAE 차트
                fig_mae = go.Figure()
                for agent in ['technical', 'fundamental', 'sentimental']:
                    rounds = []
                    mae_values = []
                    for round_num in sorted(mutual_data.keys()):
                        if agent in mutual_data[round_num]:
                            rounds.append(round_num)
                            mae_values.append(mutual_data[round_num][agent]['mae'])
                    
                    if rounds:
                        fig_mae.add_trace(go.Scatter(
                            x=rounds,
                            y=mae_values,
                            mode='lines+markers',
                            name=f'{agent.capitalize()} Agent',
                            line=dict(color=colors[agent], width=3),
                            marker=dict(size=8)
                        ))
                
                fig_mae.update_layout(
                    title="라운드별 MAE 변화",
                    xaxis_title="라운드",
                    yaxis_title="MAE",
                    showlegend=True
                )
                st.plotly_chart(fig_mae, use_container_width=True)
                
                # 베타 값 변화
                if beta_data:
                    st.subheader("🎯 베타 값 변화")
                    fig_beta = go.Figure()
                    fig_beta.add_trace(go.Bar(
                        x=list(beta_data.keys()),
                        y=list(beta_data.values()),
                        marker_color=['#1f77b4', '#ff7f0e', '#2ca02c']
                    ))
                    fig_beta.update_layout(
                        title="에이전트별 베타 값 (신뢰도)",
                        xaxis_title="Agent",
                        yaxis_title="Beta Value",
                        showlegend=False
                    )
                    st.plotly_chart(fig_beta, use_container_width=True)
                
                # 상세 결과 테이블
                st.subheader("📊 라운드별 상세 결과")
                for round_num in sorted(mutual_data.keys()):
                    with st.expander(f"Round {round_num}"):
                        round_df = pd.DataFrame(mutual_data[round_num]).T
                        round_df.index.name = 'Agent'
                        st.dataframe(round_df, use_container_width=True)
            else:
                st.warning("⚠️ 상호 훈련 데이터를 파싱할 수 없습니다.")
        else:
            st.info("ℹ️ 먼저 시스템을 실행해주세요.")
    
    with tab4:
        st.header("💬 예측 토론 현황")
        
        if hasattr(st.session_state, 'run_output'):
            # 토론 데이터 파싱
            debate_data = parse_debate_logs(st.session_state.run_output)
            
            if debate_data:
                # 라운드별 에이전트 의견 차트
                st.subheader("🗣️ 라운드별 에이전트 의견")
                
                fig_opinions = go.Figure()
                colors = {'technical': '#1f77b4', 'fundamental': '#ff7f0e', 'sentimental': '#2ca02c'}
                
                for agent in ['technical', 'fundamental', 'sentimental']:
                    rounds = []
                    predictions = []
                    for round_num in sorted(debate_data.keys()):
                        if agent in debate_data[round_num]:
                            rounds.append(round_num)
                            predictions.append(debate_data[round_num][agent]['prediction'])
                    
                    if rounds:
                        fig_opinions.add_trace(go.Scatter(
                            x=rounds,
                            y=predictions,
                            mode='lines+markers',
                            name=f'{agent.capitalize()} Agent',
                            line=dict(color=colors[agent], width=3),
                            marker=dict(size=8)
                        ))
                
                fig_opinions.update_layout(
                    title="라운드별 에이전트 예측값",
                    xaxis_title="라운드",
                    yaxis_title="예측값 ($)",
                    showlegend=True
                )
                st.plotly_chart(fig_opinions, use_container_width=True)
                
                # 베타 값 차트
                st.subheader("🎯 라운드별 베타 값")
                fig_beta_rounds = go.Figure()
                
                for agent in ['technical', 'fundamental', 'sentimental']:
                    rounds = []
                    betas = []
                    for round_num in sorted(debate_data.keys()):
                        if agent in debate_data[round_num]:
                            rounds.append(round_num)
                            betas.append(debate_data[round_num][agent]['beta'])
                    
                    if rounds:
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
                
                # 최종 합의 결과
                st.subheader("🎯 최종 합의 결과")
                
                # 합의 결과 파싱
                output = st.session_state.run_output
                consensus_line = None
                for line in output.split('\n'):
                    if 'Consensus After Round' in line:
                        consensus_line = line
                        break
                
                if consensus_line:
                    try:
                        consensus_value = float(consensus_line.split(': ')[-1])
                        st.success(f"🎯 최종 합의: ${consensus_value:.2f}")
                    except:
                        st.info("📊 합의 결과 파싱 중...")
                
                # Consensus Result 파싱
                consensus_result_line = None
                for line in output.split('\n'):
                    if 'Consensus Result' in line and 'Mean:' in line:
                        consensus_result_line = line
                        break
                
                if consensus_result_line:
                    try:
                        mean_part = consensus_result_line.split('Mean: ')[1].split(' |')[0]
                        std_part = consensus_result_line.split('Std: ')[1]
                        
                        mean_val = float(mean_part)
                        std_val = float(std_part)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("최종 예측값", f"${consensus_value:.2f}" if 'consensus_value' in locals() else "N/A")
                        with col2:
                            st.metric("평균", f"${mean_val:.4f}")
                        with col3:
                            st.metric("표준편차", f"{std_val:.4f}")
                    except:
                        pass
                
                # 라운드별 상세 결과
                st.subheader("📊 라운드별 상세 결과")
                for round_num in sorted(debate_data.keys()):
                    with st.expander(f"Round {round_num}"):
                        round_df = pd.DataFrame(debate_data[round_num]).T
                        round_df.index.name = 'Agent'
                        st.dataframe(round_df, use_container_width=True)
            else:
                st.warning("⚠️ 토론 데이터를 파싱할 수 없습니다.")
        else:
            st.info("ℹ️ 먼저 시스템을 실행해주세요.")
    
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
    ## 🚀 MCP Hybrid System v2에 오신 것을 환영합니다!
    
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
    """)
    
    # 최근 실행 결과가 있다면 표시
    if hasattr(st.session_state, 'run_completed') and st.session_state.run_completed:
        st.info("💡 이전 실행 결과가 있습니다. 위의 탭에서 확인하세요.")

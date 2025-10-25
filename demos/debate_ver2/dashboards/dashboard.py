import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os
import sys
import torch
import subprocess
import json
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 시스템 import
from core.preprocessing import build_dataset, load_csv_dataset
from core.training import pretrain_all_agents
from core.debate_engine import mutual_learning_with_individual_data
from agents.technical_agent import TechnicalAgent
from agents.fundamental_agent import FundamentalAgent
from agents.sentimental_agent import SentimentalAgent

# 로그 캡처를 위한 클래스
class StreamlitLogger:
    def __init__(self):
        self.logs = []
        self.max_logs = 1000
    
    def write(self, message):
        if message.strip():
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.logs.append(f"[{timestamp}] {message.strip()}")
            if len(self.logs) > self.max_logs:
                self.logs.pop(0)
    
    def flush(self):
        pass
    
    def get_logs(self):
        return self.logs
    
    def clear(self):
        self.logs = []

# 전역 로거 인스턴스
if 'logger' not in st.session_state:
    st.session_state.logger = StreamlitLogger()

def log_msg(message):
    """로그 메시지를 세션 상태에 추가"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    st.session_state.logger.logs.append(log_entry)
    if len(st.session_state.logger.logs) > 1000:
        st.session_state.logger.logs.pop(0)

# --------------------------------------------
# 데이터 수집 단계
# --------------------------------------------
def collect_data_stage(ticker):
    """Stage 0: 데이터 수집"""
    log_msg(f"📊 데이터 수집 시작: {ticker}")
    
    try:
        # CSV에서 데이터 로드 시도
        try:
            X, y, scaler_X, scaler_y, feature_cols = load_csv_dataset(ticker)
            data_info = {
                "processed_data": X,
                "targets": y,
                "scaler_X": scaler_X,
                "scaler_y": scaler_y,
                "feature_cols": feature_cols
            }
            log_msg(f"✅ CSV에서 데이터 로드 완료: {X.shape}")
        except FileNotFoundError:
            log_msg("📥 CSV 파일이 없어서 새로 데이터를 수집합니다...")
            data_info = build_dataset(ticker)
            log_msg(f"✅ 새 데이터 수집 완료: {data_info['processed_data'].shape}")
        
        # 날짜 범위 처리
        raw_df = pd.read_csv(f"data/processed/{ticker}_raw_data.csv", index_col=0, parse_dates=True)
        min_date = raw_df.index.min()
        max_date = raw_df.index.max()
        
        # 날짜가 이미 datetime 객체인지 확인
        if hasattr(min_date, 'strftime'):
            date_range = f"{min_date.strftime('%Y-%m-%d')} ~ {max_date.strftime('%Y-%m-%d')}"
        else:
            date_range = f"{str(min_date)[:10]} ~ {str(max_date)[:10]}"
        
        st.session_state.data_info = data_info
        st.session_state.data_info["date_range"] = date_range
        st.session_state.data_info["ticker"] = ticker
        
        log_msg("✅ 데이터 수집 완료")
        return True
        
    except Exception as e:
        log_msg(f"❌ 데이터 수집 실패: {e}")
        return False

# --------------------------------------------
# 사전학습 단계
# --------------------------------------------
def pretraining_stage(epochs, ticker):
    """Stage 1: 사전학습"""
    log_msg("🧠 사전학습 시작")
    
    try:
        # Agent 초기화
        agents = {
            "technical": TechnicalAgent("TechnicalAgent"),
            "fundamental": FundamentalAgent("FundamentalAgent"),
            "sentimental": SentimentalAgent("SentimentalAgent"),
        }
        
        # 각 에이전트별로 적절한 데이터셋 준비
        datasets = {}
        scalers = {}
        for name, agent in agents.items():
            try:
                X, y, scaler_X, scaler_y, _ = load_csv_dataset(ticker, name)
                X_t = torch.tensor(X, dtype=torch.float32)
                y_t = torch.tensor(y, dtype=torch.float32)
                datasets[name] = (X_t, y_t)
                scalers[name] = scaler_y
                log_msg(f"✅ {name} 데이터셋 로드: {X_t.shape}")
            except FileNotFoundError:
                log_msg(f"⚠️ {name} 데이터셋 없음, 기본 데이터 사용")
                # 기본 데이터 사용 (모든 에이전트가 공통으로 사용)
                data_info = st.session_state.data_info
                X, y = data_info["processed_data"], data_info["targets"]
                X_t, y_t = torch.tensor(X), torch.tensor(y)
                datasets[name] = (X_t, y_t)
                scalers[name] = data_info.get("scaler_y", None)
        
        # Agent 정보 저장
        agents_info = {}
        for name, agent in agents.items():
            agents_info[name] = {
                "agent_id": agent.agent_id,
                "parameters": sum(p.numel() for p in agent.parameters() if p.requires_grad),
                "model_type": type(agent).__name__
            }
        
        st.session_state.agents_info = agents_info
        
        # 사전학습
        pretrain_all_agents(agents, datasets, epochs=epochs, save_models=True)
        
        # 학습 후 예측 테스트 (실제 주가와 비교 가능하도록)
        predictions = {}
        actual_prices = {}
        
        for name, agent in agents.items():
            X_agent, y_agent = datasets[name]
            with torch.no_grad():
                # 스케일링된 예측값
                preds_scaled = agent.forward(X_agent[:10]).detach().numpy().flatten()
                predictions[name] = preds_scaled
                
                # 실제 주가 데이터로 예측 (스케일링 해제된 값)
                try:
                    stock_data = agent.searcher(ticker)
                    target = agent.predicter(stock_data)
                    # 정규화된 예측값을 실제 주가로 변환
                    if name in scalers and scalers[name] is not None:
                        scaler_y = scalers[name]
                        # 정규화된 값을 실제 주가로 역변환
                        normalized_pred = np.array([[target.next_close]])
                        actual_price = scaler_y.inverse_transform(normalized_pred)[0][0]
                        log_msg(f"🔍 {name} Agent: 정규화값={target.next_close:.6f} → 실제주가={actual_price:.2f}")
                        actual_prices[name] = actual_price
                    else:
                        log_msg(f"⚠️ {name} Agent: 스케일러 없음, 정규화값 그대로 사용={target.next_close:.6f}")
                        actual_prices[name] = target.next_close
                except Exception as e:
                    log_msg(f"⚠️ {name} Agent 실제 예측 실패: {e}")
                    actual_prices[name] = 100.0
        
        st.session_state.pipeline_results["pretraining"] = {
            "agents": agents,
            "predictions": predictions,
            "actual_prices": actual_prices,
            "epochs": epochs
        }
        
        log_msg("✅ 사전학습 완료")
        return True
        
    except Exception as e:
        log_msg(f"❌ 사전학습 실패: {e}")
        return False

    # --------------------------------------------
# 상호학습 단계
# --------------------------------------------
def mutual_learning_stage(rounds, ticker):
    """Stage 2: 상호학습"""
    log_msg("🔁 상호학습 시작")
    
    try:
        agents = st.session_state.pipeline_results["pretraining"]["agents"]
        
        # 각 에이전트별로 적절한 데이터셋 준비
        datasets = {}
        scalers = {}
        for name, agent in agents.items():
            try:
                X, y, scaler_X, scaler_y, _ = load_csv_dataset(ticker, name)
                X_t = torch.tensor(X, dtype=torch.float32)
                y_t = torch.tensor(y, dtype=torch.float32)
                datasets[name] = (X_t, y_t)
                scalers[name] = scaler_y
                log_msg(f"✅ {name} 데이터셋 로드: {X_t.shape}")
            except FileNotFoundError:
                log_msg(f"⚠️ {name} 데이터셋 없음, 기본 데이터 사용")
                # 기본 데이터 사용 (모든 에이전트가 공통으로 사용)
                data_info = st.session_state.data_info
                X, y = data_info["processed_data"], data_info["targets"]
                X_t, y_t = torch.tensor(X), torch.tensor(y)
                datasets[name] = (X_t, y_t)
                scalers[name] = data_info.get("scaler_y", None)
        
        # 상호학습 전 예측값 저장
        initial_predictions = {}
        initial_actual_prices = {}
        for name, agent in agents.items():
            X_agent, y_agent = datasets[name]
            with torch.no_grad():
                preds = agent.forward(X_agent).detach().numpy()
                initial_predictions[name] = preds.flatten()
                
                try:
                    stock_data = agent.searcher(ticker)
                    target = agent.predicter(stock_data)
                    # 정규화된 예측값을 실제 주가로 변환
                    if name in scalers and scalers[name] is not None:
                        scaler_y = scalers[name]
                        normalized_pred = np.array([[target.next_close]])
                        actual_price = scaler_y.inverse_transform(normalized_pred)[0][0]
                        initial_actual_prices[name] = actual_price
                    else:
                        initial_actual_prices[name] = target.next_close
                except Exception as e:
                    log_msg(f"⚠️ {name} Agent 초기 실제 예측 실패: {e}")
                    initial_actual_prices[name] = 100.0
        
        # 상호학습 실행
        mutual_learning_with_individual_data(agents, datasets, rounds=rounds, save_models=True)
        
        # 상호학습 후 예측값 저장
        final_predictions = {}
        final_actual_prices = {}
        for name, agent in agents.items():
            X_agent, y_agent = datasets[name]
            with torch.no_grad():
                preds = agent.forward(X_agent).detach().numpy()
                final_predictions[name] = preds.flatten()
                
                try:
                    stock_data = agent.searcher(ticker)
                    target = agent.predicter(stock_data)
                    # 정규화된 예측값을 실제 주가로 변환
                    if name in scalers and scalers[name] is not None:
                        scaler_y = scalers[name]
                        normalized_pred = np.array([[target.next_close]])
                        actual_price = scaler_y.inverse_transform(normalized_pred)[0][0]
                        final_actual_prices[name] = actual_price
                    else:
                        final_actual_prices[name] = target.next_close
                except Exception as e:
                    log_msg(f"⚠️ {name} Agent 최종 실제 예측 실패: {e}")
                    final_actual_prices[name] = 100.0
        
        st.session_state.pipeline_results["mutual_learning"] = {
            "agents": agents,
            "initial_predictions": initial_predictions,
            "final_predictions": final_predictions,
            "initial_actual_prices": initial_actual_prices,
            "final_actual_prices": final_actual_prices,
            "rounds": rounds
        }
        
        log_msg("✅ 상호학습 완료")
        return True
        
    except Exception as e:
        log_msg(f"❌ 상호학습 실패: {e}")
        return False

# --------------------------------------------
# 디베이트 단계
    # --------------------------------------------
def debate_stage(ticker, rounds):
    """Stage 3: 디베이트"""
    log_msg("💬 디베이트 시작")
    
    try:
        agents = st.session_state.pipeline_results["mutual_learning"]["agents"]
        data_info = st.session_state.data_info
        
        # 각 에이전트별 스케일러 준비
        scalers = {}
        for name, agent in agents.items():
            try:
                _, _, _, scaler_y, _ = load_csv_dataset(ticker, name)
                scalers[name] = scaler_y
            except FileNotFoundError:
                scalers[name] = data_info.get("scaler_y", None)
        
        # 라운드별 결과 수집
        debate_results = []
        for round_num in range(1, rounds + 1):
            log_msg(f"💬 디베이트 라운드 {round_num} 시작")
            
            round_predictions = {}
            for name, agent in agents.items():
                try:
                    stock_data = agent.searcher(ticker)
                    target = agent.predicter(stock_data)
                    # 정규화된 예측값을 실제 주가로 변환
                    if hasattr(st.session_state, 'data_info') and 'scaler_y' in st.session_state.data_info:
                        scaler_y = st.session_state.data_info['scaler_y']
                        normalized_pred = np.array([[target.next_close]])
                        actual_price = scaler_y.inverse_transform(normalized_pred)[0][0]
                        round_predictions[name] = actual_price
                    else:
                        round_predictions[name] = target.next_close
                except Exception as e:
                    log_msg(f"⚠️ {name} Agent 예측 실패: {e}")
                    round_predictions[name] = 100.0
            
            round_consensus = sum(round_predictions.values()) / len(round_predictions)
            debate_results.append({
                "round": round_num,
                "predictions": round_predictions,
                "consensus": round_consensus
            })
            log_msg(f"📊 라운드 {round_num} 합의: {round_consensus:.2f}")
        
        # 최종 평가 (스케일링된 데이터 사용)
        X, y = data_info["processed_data"], data_info["targets"]
        X_t, y_t = torch.tensor(X), torch.tensor(y)
        
        evaluation_results = {}
        for name, agent in agents.items():
            try:
                with torch.no_grad():
                    preds = agent.forward(X_t).detach().numpy().flatten()
                    y_true = y_t.numpy().flatten()
                    
                    mse = np.mean((preds - y_true) ** 2)
                    rmse = np.sqrt(mse)
                    mae = np.mean(np.abs(preds - y_true))
                    ss_res = np.sum((y_true - preds) ** 2)
                    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                    r2 = 1 - (ss_res / (ss_tot + 1e-8))
                    
                    evaluation_results[name] = {
                        "RMSE": rmse,
                        "MAE": mae,
                        "R2": r2,
                        "MSE": mse
                    }
            except Exception as e:
                log_msg(f"⚠️ {name} Agent 평가 실패: {e}")
                evaluation_results[name] = {
                    "RMSE": 0.0, "MAE": 0.0, "R2": 0.0, "MSE": 0.0
                }
        
        final_consensus = {
            "mean": debate_results[-1]["consensus"] if debate_results else 0.0,
            "std": np.std([r["consensus"] for r in debate_results]) if debate_results else 0.0,
            "rounds": rounds
        }
        
        st.session_state.pipeline_results["debate"] = {
            "agents": agents,
            "evaluation_results": evaluation_results,
            "consensus": final_consensus,
            "round_results": debate_results,
            "rounds": rounds
        }
        
        log_msg("✅ 디베이트 완료")
        return True
        
    except Exception as e:
        log_msg(f"❌ 디베이트 실패: {e}")
        return False

# --------------------------------------------
# 메인 대시보드
# --------------------------------------------
def main():
    st.set_page_config(
        page_title="MCP Hybrid System Dashboard",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 MCP Hybrid System Dashboard")
    st.markdown("**Multi-Agent Collaborative Prediction System**")
    
    # 사이드바 설정
    st.sidebar.header("🎛️ 시스템 설정")
    
    with st.sidebar.expander("ℹ️ 사용법"):
        st.markdown("""
        **MCP Hybrid System 사용법:**
        
        1. **Ticker**: 분석할 종목 코드 입력
        2. **Epochs**: 사전학습 반복 횟수 (높을수록 정확하지만 시간 소요)
        3. **Mutual Rounds**: 상호학습 라운드 수
        4. **Debate Rounds**: LLM 토론 라운드 수
        5. **분석 시작** 버튼 클릭
        
        **권장 설정:**
        - 빠른 테스트: Epochs=5, Rounds=1
        - 정확한 분석: Epochs=20, Rounds=3
        """)
    
    ticker = st.sidebar.text_input("Ticker Symbol", "TSLA", help="예: TSLA, AAPL, MSFT")
    epochs = st.sidebar.slider("Pretraining Epochs", 5, 50, 20, help="사전학습 반복 횟수")
    mutual_rounds = st.sidebar.slider("Mutual Learning Rounds", 1, 10, 3, help="상호학습 라운드 수")
    debate_rounds = st.sidebar.slider("Debate Rounds", 1, 5, 2, help="LLM 토론 라운드 수")
    
    run_button = st.sidebar.button("🚀 분석 시작", type="primary")
    
    # 세션 상태 초기화
    if "pipeline_results" not in st.session_state:
        st.session_state.pipeline_results = {}
    
    # 파이프라인 실행
    if run_button:
        st.session_state.pipeline_results = {}
        st.session_state.logger.clear()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Stage 0: 데이터 수집
        status_text.text("📊 데이터 수집 중...")
        progress_bar.progress(10)
        if collect_data_stage(ticker):
            progress_bar.progress(25)
            
            # Stage 1: 사전학습
            status_text.text("🧠 사전학습 중...")
            if pretraining_stage(epochs, ticker):
                progress_bar.progress(50)
                
                # Stage 2: 상호학습
                status_text.text("🔁 상호학습 중...")
                if mutual_learning_stage(mutual_rounds, ticker):
                    progress_bar.progress(75)
                    
                    # Stage 3: 디베이트
                    status_text.text("💬 디베이트 중...")
                    if debate_stage(ticker, debate_rounds):
                        progress_bar.progress(100)
                        status_text.text("✅ 모든 단계 완료!")
                    else:
                        status_text.text("❌ 디베이트 실패")
                else:
                    status_text.text("❌ 상호학습 실패")
            else:
                status_text.text("❌ 사전학습 실패")
        else:
            status_text.text("❌ 데이터 수집 실패")
    
    # 로그 표시
    with st.sidebar.expander("🧾 실행 로그", expanded=False):
        logs = st.session_state.logger.get_logs()
        if logs:
            # 최근 10개 로그만 표시
            recent_logs = logs[-10:] if len(logs) > 10 else logs
            for log in recent_logs:
                st.text(log)
        else:
            st.text("로그가 없습니다.")
    
    # 탭 구성
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["데이터 수집", "사전학습", "상호학습", "디베이트", "종합 결과"])
    
    # 데이터 수집 탭
    with tab1:
        st.header("📊 데이터 수집 결과")
        
        if "data_info" in st.session_state:
            data_info = st.session_state.data_info
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Ticker", data_info.get("ticker", "N/A"))
            with col2:
                st.metric("데이터 포인트", f"{data_info['processed_data'].shape[0]:,}")
            with col3:
                st.metric("피처 수", data_info['processed_data'].shape[1])
            
            if "date_range" in data_info:
                st.info(f"📅 데이터 기간: {data_info['date_range']}")
            
            # 데이터 미리보기
            st.subheader("📋 데이터 미리보기")
            raw_df = pd.read_csv(f"data/processed/{data_info.get('ticker', 'TSLA')}_raw_data.csv", index_col=0, parse_dates=True)
            st.dataframe(raw_df.head(10), width='stretch')
            
            # 가격 차트
            st.subheader("📈 가격 차트")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=raw_df.index,
                y=raw_df['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='blue', width=2)
            ))
            fig.update_layout(
                title=f"{data_info.get('ticker', 'TSLA')} 주가 차트",
                xaxis_title="날짜",
                yaxis_title="가격 ($)",
                height=400
            )
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("데이터를 수집하려면 '분석 시작' 버튼을 클릭하세요.")
    
    # 사전학습 탭
    with tab2:
        st.header("🧠 사전학습 결과")
        
        if "pretraining" in st.session_state.pipeline_results:
            pretraining_data = st.session_state.pipeline_results["pretraining"]
            
            # Agent 정보
            if "agents_info" in st.session_state:
                st.subheader("🤖 Agent 정보")
                agents_info = st.session_state.agents_info
                
                for name, info in agents_info.items():
                    with st.expander(f"{info['agent_id']} ({info['model_type']})"):
                        st.write(f"**파라미터 수**: {info['parameters']:,}")
                        st.write(f"**모델 타입**: {info['model_type']}")
            
            # 예측 결과 비교
            st.subheader("📊 예측 결과 비교")
            
            if "actual_prices" in pretraining_data:
                actual_prices = pretraining_data["actual_prices"]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**실제 주가 예측값**")
                    for name, price in actual_prices.items():
                        st.metric(f"{name.title()} Agent", f"${price:.2f}")
                
                with col2:
                    st.write("**스케일링된 예측값**")
                    predictions = pretraining_data["predictions"]
                    for name, preds in predictions.items():
                        avg_pred = np.mean(preds)
                        st.metric(f"{name.title()} Agent", f"{avg_pred:.4f}")
            
            # 학습 진행도
            st.subheader("📈 학습 진행도")
            epochs = pretraining_data.get("epochs", 0)
            st.info(f"✅ {epochs} 에포크 사전학습 완료")
            
        else:
            st.info("사전학습을 실행하려면 '분석 시작' 버튼을 클릭하세요.")
    
    # 상호학습 탭
    with tab3:
        st.header("🔁 상호학습 결과")
        
        if "mutual_learning" in st.session_state.pipeline_results:
            mutual_data = st.session_state.pipeline_results["mutual_learning"]
            
            # 상호학습 전후 비교
            st.subheader("📊 상호학습 전후 비교")
            
            if "initial_actual_prices" in mutual_data and "final_actual_prices" in mutual_data:
                initial_prices = mutual_data["initial_actual_prices"]
                final_prices = mutual_data["final_actual_prices"]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**상호학습 전**")
                    for name, price in initial_prices.items():
                        st.metric(f"{name.title()} Agent", f"${price:.2f}")
                
                with col2:
                    st.write("**상호학습 후**")
                    for name, price in final_prices.items():
                        st.metric(f"{name.title()} Agent", f"${price:.2f}")
                
                with col3:
                    st.write("**변화량**")
                    for name in initial_prices.keys():
                        change = final_prices[name] - initial_prices[name]
                        st.metric(f"{name.title()} Agent", f"{change:+.2f}")
            
            # 상호학습 진행도
            st.subheader("📈 상호학습 진행도")
            rounds = mutual_data.get("rounds", 0)
            st.info(f"✅ {rounds} 라운드 상호학습 완료")
            
        else:
            st.info("상호학습을 실행하려면 '분석 시작' 버튼을 클릭하세요.")
    
    # 디베이트 탭
    with tab4:
        st.header("💬 디베이트 결과")
        
        if "debate" in st.session_state.pipeline_results:
            debate_data = st.session_state.pipeline_results["debate"]
            
            # 라운드별 결과
            if "round_results" in debate_data:
                st.subheader("📊 라운드별 디베이트 결과")
                
                round_results = debate_data["round_results"]
                
                # 라운드별 차트
                rounds = [r["round"] for r in round_results]
                consensus_values = [r["consensus"] for r in round_results]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=rounds,
                    y=consensus_values,
                    mode='lines+markers',
                    name='Consensus',
                    line=dict(color='red', width=3),
                    marker=dict(size=8)
                ))
                
                fig.update_layout(
                    title="라운드별 합의 가격",
                    xaxis_title="라운드",
                    yaxis_title="합의 가격 ($)",
                    height=400
                )
                st.plotly_chart(fig, width='stretch')
                
                # 라운드별 상세 결과 테이블
                st.subheader("📋 라운드별 상세 결과")
                round_data = []
                for result in round_results:
                    row = {"라운드": result["round"], "합의": f"${result['consensus']:.2f}"}
                    for name, pred in result["predictions"].items():
                        row[f"{name.title()} Agent"] = f"${pred:.2f}"
                    round_data.append(row)
                
                round_df = pd.DataFrame(round_data)
                st.dataframe(round_df, width='stretch')
            
            # 성능 평가
            if "evaluation_results" in debate_data:
                st.subheader("📈 성능 평가")
                
                evaluation_results = debate_data["evaluation_results"]
                
                col1, col2, col3 = st.columns(3)
                
                for i, (name, metrics) in enumerate(evaluation_results.items()):
                    with [col1, col2, col3][i]:
                        st.write(f"**{name.title()} Agent**")
                        st.metric("RMSE", f"{metrics['RMSE']:.4f}")
                        st.metric("MAE", f"{metrics['MAE']:.4f}")
                        st.metric("R²", f"{metrics['R2']:.4f}")
            
            # 최종 합의
            if "consensus" in debate_data:
                st.subheader("🎯 최종 합의")
                consensus = debate_data["consensus"]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("평균 합의", f"${consensus['mean']:.2f}")
                with col2:
                    st.metric("표준편차", f"${consensus['std']:.2f}")
            
        else:
            st.info("디베이트를 실행하려면 '분석 시작' 버튼을 클릭하세요.")
    
    # 종합 결과 탭
    with tab5:
        st.header("🎯 종합 결과")
        
        if st.session_state.pipeline_results:
            st.success("✅ 모든 단계가 성공적으로 완료되었습니다!")
            
            # 전체 파이프라인 요약
            st.subheader("📊 파이프라인 요약")
            
            if "data_info" in st.session_state:
                data_info = st.session_state.data_info
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Ticker", data_info.get("ticker", "N/A"))
                with col2:
                    st.metric("데이터 포인트", f"{data_info['processed_data'].shape[0]:,}")
                with col3:
                    st.metric("피처 수", data_info['processed_data'].shape[1])
                with col4:
                    if "debate" in st.session_state.pipeline_results:
                        consensus = st.session_state.pipeline_results["debate"]["consensus"]
                        st.metric("최종 합의", f"${consensus['mean']:.2f}")
            
            # 최종 예측값 비교
            if "debate" in st.session_state.pipeline_results and "round_results" in st.session_state.pipeline_results["debate"]:
                st.subheader("🎯 최종 예측값")
                
                final_round = st.session_state.pipeline_results["debate"]["round_results"][-1]
                final_predictions = final_round["predictions"]
                final_consensus = final_round["consensus"]
                
                col1, col2, col3, col4 = st.columns(4)
                
                for i, (name, pred) in enumerate(final_predictions.items()):
                    with [col1, col2, col3][i]:
                        st.metric(f"{name.title()} Agent", f"${pred:.2f}")
                
                with col4:
                    st.metric("최종 합의", f"${final_consensus:.2f}", delta=f"${final_consensus - np.mean(list(final_predictions.values())):+.2f}")
            
        else:
            st.info("종합 결과를 보려면 '분석 시작' 버튼을 클릭하세요.")

if __name__ == "__main__":
    main()

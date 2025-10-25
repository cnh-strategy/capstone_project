import streamlit as st
import subprocess
import json
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# 페이지 설정
st.set_page_config(
    page_title="MCP Hybrid System Dashboard",
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
</style>
""", unsafe_allow_html=True)

# 메인 헤더
st.markdown('<h1 class="main-header">🤖 MCP Hybrid System Dashboard</h1>', unsafe_allow_html=True)

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
        "🎯 훈련 과정", 
        "🔄 상호 훈련", 
        "💬 예측 토론", 
        "📝 실행 로그"
    ])
    
    with tab1:
        st.header("📊 데이터 수집 결과")
        
        # 데이터 파일 확인
        data_dir = f"/home/ubuntu/Projects/ml-ai/capstone/demos/debate_ver2/data/processed"
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Technical Agent")
            tech_file = f"{data_dir}/{ticker}_technical_dataset.csv"
            if os.path.exists(tech_file):
                df = pd.read_csv(tech_file)
                st.success(f"✅ 데이터 로드 완료")
                st.metric("샘플 수", len(df))
                st.metric("피처 수", len(df.columns) - 1)  # Close 컬럼 제외
                
                # 데이터 미리보기
                with st.expander("📊 데이터 미리보기"):
                    st.dataframe(df.head(), use_container_width=True)
                
                # 통계 정보
                with st.expander("📈 통계 정보"):
                    st.write("**컬럼 목록:**")
                    st.write(list(df.columns))
                    if 'Close' in df.columns:
                        st.metric("평균 종가", f"${df['Close'].mean():.2f}")
                        st.metric("최고가", f"${df['Close'].max():.2f}")
                        st.metric("최저가", f"${df['Close'].min():.2f}")
            else:
                st.error("❌ 데이터 파일 없음")
        
        with col2:
            st.subheader("Fundamental Agent")
            fund_file = f"{data_dir}/{ticker}_fundamental_dataset.csv"
            if os.path.exists(fund_file):
                df = pd.read_csv(fund_file)
                st.success(f"✅ 데이터 로드 완료")
                st.metric("샘플 수", len(df))
                st.metric("피처 수", len(df.columns) - 1)
                
                # 데이터 미리보기
                with st.expander("📊 데이터 미리보기"):
                    st.dataframe(df.head(), use_container_width=True)
                
                # 통계 정보
                with st.expander("📈 통계 정보"):
                    st.write("**컬럼 목록:**")
                    st.write(list(df.columns))
                    if 'Close' in df.columns:
                        st.metric("평균 종가", f"${df['Close'].mean():.2f}")
                        st.metric("최고가", f"${df['Close'].max():.2f}")
                        st.metric("최저가", f"${df['Close'].min():.2f}")
            else:
                st.error("❌ 데이터 파일 없음")
        
        with col3:
            st.subheader("Sentimental Agent")
            sent_file = f"{data_dir}/{ticker}_sentimental_dataset.csv"
            if os.path.exists(sent_file):
                df = pd.read_csv(sent_file)
                st.success(f"✅ 데이터 로드 완료")
                st.metric("샘플 수", len(df))
                st.metric("피처 수", len(df.columns) - 1)
                
                # 데이터 미리보기
                with st.expander("📊 데이터 미리보기"):
                    st.dataframe(df.head(), use_container_width=True)
                
                # 통계 정보
                with st.expander("📈 통계 정보"):
                    st.write("**컬럼 목록:**")
                    st.write(list(df.columns))
                    if 'Close' in df.columns:
                        st.metric("평균 종가", f"${df['Close'].mean():.2f}")
                        st.metric("최고가", f"${df['Close'].max():.2f}")
                        st.metric("최저가", f"${df['Close'].min():.2f}")
            else:
                st.error("❌ 데이터 파일 없음")
    
    with tab2:
        st.header("🎯 훈련 과정 결과")
        
        # 모델 파일 확인
        models_dir = f"/home/ubuntu/Projects/ml-ai/capstone/demos/debate_ver2/models"
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Technical Agent")
            tech_model = f"{models_dir}/technical_agent.pt"
            if os.path.exists(tech_model):
                size = os.path.getsize(tech_model) / 1024  # KB
                st.success(f"✅ 모델 훈련 완료")
                st.metric("모델 크기", f"{size:.1f} KB")
            else:
                st.error("❌ 모델 파일 없음")
        
        with col2:
            st.subheader("Fundamental Agent")
            fund_model = f"{models_dir}/fundamental_agent.pt"
            if os.path.exists(fund_model):
                size = os.path.getsize(fund_model) / 1024
                st.success(f"✅ 모델 훈련 완료")
                st.metric("모델 크기", f"{size:.1f} KB")
            else:
                st.error("❌ 모델 파일 없음")
        
        with col3:
            st.subheader("Sentimental Agent")
            sent_model = f"{models_dir}/sentimental_agent.pt"
            if os.path.exists(sent_model):
                size = os.path.getsize(sent_model) / 1024
                st.success(f"✅ 모델 훈련 완료")
                st.metric("모델 크기", f"{size:.1f} KB")
            else:
                st.error("❌ 모델 파일 없음")
    
    with tab3:
        st.header("🔄 상호 훈련 결과")
        
        # 상호학습된 모델 확인
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Technical Agent")
            tech_finetuned = f"{models_dir}/technical_agent_finetuned.pt"
            if os.path.exists(tech_finetuned):
                size = os.path.getsize(tech_finetuned) / 1024
                st.success(f"✅ 상호학습 완료")
                st.metric("모델 크기", f"{size:.1f} KB")
            else:
                st.error("❌ 상호학습 모델 없음")
        
        with col2:
            st.subheader("Fundamental Agent")
            fund_finetuned = f"{models_dir}/fundamental_agent_finetuned.pt"
            if os.path.exists(fund_finetuned):
                size = os.path.getsize(fund_finetuned) / 1024
                st.success(f"✅ 상호학습 완료")
                st.metric("모델 크기", f"{size:.1f} KB")
            else:
                st.error("❌ 상호학습 모델 없음")
        
        with col3:
            st.subheader("Sentimental Agent")
            sent_finetuned = f"{models_dir}/sentimental_agent_finetuned.pt"
            if os.path.exists(sent_finetuned):
                size = os.path.getsize(sent_finetuned) / 1024
                st.success(f"✅ 상호학습 완료")
                st.metric("모델 크기", f"{size:.1f} KB")
            else:
                st.error("❌ 상호학습 모델 없음")
        
        # 성능 비교 차트 (실제 데이터)
        st.subheader("📈 성능 비교")
        
        # run.py 출력에서 실제 성능 데이터 파싱
        output = st.session_state.run_output
        
        # MSE/MAE 결과 파싱
        performance_data = {
            'Agent': ['Technical', 'Fundamental', 'Sentimental'],
            'MSE': [0.0, 0.0, 0.0],
            'MAE': [0.0, 0.0, 0.0]
        }
        
        for line in output.split('\n'):
            if 'MSE=' in line and 'MAE=' in line:
                parts = line.split(': ')
                if len(parts) > 1:
                    agent_name = parts[0].strip().replace('   - ', '').lower()
                    metrics = parts[1].strip()
                    
                    # MSE와 MAE 값 추출
                    mse_match = metrics.split('MSE=')[1].split(',')[0]
                    mae_match = metrics.split('MAE=')[1]
                    
                    try:
                        mse_val = float(mse_match)
                        mae_val = float(mae_match)
                        
                        if agent_name == 'technical':
                            performance_data['MSE'][0] = mse_val
                            performance_data['MAE'][0] = mae_val
                        elif agent_name == 'fundamental':
                            performance_data['MSE'][1] = mse_val
                            performance_data['MAE'][1] = mae_val
                        elif agent_name == 'sentimental':
                            performance_data['MSE'][2] = mse_val
                            performance_data['MAE'][2] = mae_val
                    except:
                        pass
        
        df_perf = pd.DataFrame(performance_data)
        
        # MSE 차트
        fig_mse = go.Figure()
        fig_mse.add_trace(go.Bar(x=df_perf['Agent'], y=df_perf['MSE'], name='MSE'))
        fig_mse.update_layout(
            title="에이전트별 MSE (Mean Squared Error)",
            xaxis_title="Agent",
            yaxis_title="MSE",
            showlegend=False
        )
        
        # MAE 차트
        fig_mae = go.Figure()
        fig_mae.add_trace(go.Bar(x=df_perf['Agent'], y=df_perf['MAE'], name='MAE'))
        fig_mae.update_layout(
            title="에이전트별 MAE (Mean Absolute Error)",
            xaxis_title="Agent",
            yaxis_title="MAE",
            showlegend=False
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_mse, use_container_width=True)
        with col2:
            st.plotly_chart(fig_mae, use_container_width=True)
        
        # 성능 테이블
        st.subheader("📊 성능 지표 상세")
        st.dataframe(df_perf, use_container_width=True)
    
    with tab4:
        st.header("💬 예측 토론 결과")
        
        # 실행 로그에서 최종 합의 결과 파싱
        output = st.session_state.run_output
        
        # 합의 결과 추출
        consensus_line = None
        for line in output.split('\n'):
            if 'Consensus After Round' in line:
                consensus_line = line
                break
        
        if consensus_line:
            # 합의 값 추출
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
                # "Mean: 3.6133 | Std: 2.2254" 형태에서 값 추출
                mean_part = consensus_result_line.split('Mean: ')[1].split(' |')[0]
                std_part = consensus_result_line.split('Std: ')[1]
                
                mean_val = float(mean_part)
                std_val = float(std_part)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("평균 예측값", f"${mean_val:.4f}")
                with col2:
                    st.metric("표준편차", f"{std_val:.4f}")
            except:
                pass
        
        # 평가 결과 표시
        st.subheader("📊 에이전트별 평가")
        
        # MSE/MAE 결과 파싱
        evaluation_results = []
        for line in output.split('\n'):
            if 'MSE=' in line and 'MAE=' in line:
                parts = line.split(': ')
                if len(parts) > 1:
                    agent_name = parts[0].strip().replace('   - ', '')
                    metrics = parts[1].strip()
                    evaluation_results.append({
                        'Agent': agent_name,
                        'Metrics': metrics
                    })
        
        if evaluation_results:
            # 테이블 형태로 표시
            eval_df = pd.DataFrame(evaluation_results)
            st.dataframe(eval_df, use_container_width=True)
            
            # 개별 메트릭으로도 표시
            st.subheader("📈 개별 성능 지표")
            for result in evaluation_results:
                st.metric(result['Agent'], result['Metrics'])
        else:
            st.info("📊 평가 결과를 분석 중...")
        
        # 토론 라운드별 결과 파싱
        st.subheader("🔄 토론 라운드별 결과")
        
        round_results = []
        current_round = None
        
        for line in output.split('\n'):
            if 'Round' in line and 'Debug:' in line:
                # 라운드 번호 추출
                try:
                    round_num = int(line.split('Round ')[1].split()[0])
                    current_round = round_num
                except:
                    pass
            elif 'Consensus After Round' in line and current_round:
                try:
                    consensus_val = float(line.split(': ')[-1])
                    round_results.append({
                        'Round': current_round,
                        'Consensus': consensus_val
                    })
                except:
                    pass
        
        if round_results:
            round_df = pd.DataFrame(round_results)
            
            # 라운드별 합의 차트
            fig_rounds = go.Figure()
            fig_rounds.add_trace(go.Scatter(
                x=round_df['Round'], 
                y=round_df['Consensus'],
                mode='lines+markers',
                name='합의값',
                line=dict(color='blue', width=3),
                marker=dict(size=10)
            ))
            
            fig_rounds.update_layout(
                title="라운드별 합의값 변화",
                xaxis_title="라운드",
                yaxis_title="합의값 ($)",
                showlegend=False
            )
            
            st.plotly_chart(fig_rounds, use_container_width=True)
            st.dataframe(round_df, use_container_width=True)
        else:
            st.info("📊 라운드별 결과를 분석 중...")
    
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
    ## 🚀 MCP Hybrid System에 오신 것을 환영합니다!
    
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

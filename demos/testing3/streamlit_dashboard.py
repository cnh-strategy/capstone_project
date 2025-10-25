#!/usr/bin/env python3
"""
새로운 하이브리드 주식 예측 시스템 - Streamlit 대시보드
원래 capstone 구조를 기반으로 한 깔끔한 UI
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 시스템 import
from main import HybridStockPredictionSystem

# 페이지 설정
st.set_page_config(
    page_title="🚀 하이브리드 주식 예측 시스템",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 사이드바
st.sidebar.title("🎯 하이브리드 주식 예측 시스템")
st.sidebar.markdown("---")

# 입력 설정
ticker = st.sidebar.text_input(
    "📈 주식 티커",
    value="RZLV",
    help="분석할 주식 티커를 입력하세요 (예: RZLV, AAPL, TSLA)"
).upper()

# 분석 단계 선택
st.sidebar.markdown("### 🔄 분석 단계")
step_options = {
    "전체 분석": "all",
    "데이터 수집만": "search", 
    "모델 학습만": "train",
    "예측만": "predict",
    "토론만": "debate"
}

selected_step = st.sidebar.selectbox(
    "실행할 단계 선택",
    list(step_options.keys()),
    help="분석할 단계를 선택하세요"
)

# 고급 설정
with st.sidebar.expander("🔧 고급 설정"):
    force_retrain = st.checkbox(
        "모델 강제 재학습",
        value=False,
        help="기존 모델을 무시하고 새로 학습합니다"
    )
    
    debate_rounds = st.slider(
        "토론 라운드 수",
        min_value=1,
        max_value=5,
        value=3,
        help="LLM 토론에서 진행할 라운드 수"
    )

# 분석 실행 버튼
if st.sidebar.button("🚀 분석 시작", type="primary"):
    if not ticker:
        st.error("❌ 주식 티커를 입력해주세요.")
    else:
        # 진행 상황 표시
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # 시스템 초기화
            status_text.text("🔄 시스템 초기화 중...")
            progress_bar.progress(10)
            
            system = HybridStockPredictionSystem(ticker)
            
            # 선택된 단계 실행
            if selected_step == "전체 분석":
                status_text.text("📊 전체 분석 실행 중...")
                progress_bar.progress(30)
                
                results = system.run_full_analysis(force_retrain, debate_rounds)
                
                # 결과를 세션 상태에 저장
                st.session_state.analysis_results = results
                
            elif selected_step == "데이터 수집만":
                status_text.text("🔍 데이터 수집 중...")
                progress_bar.progress(50)
                
                results = system.step1_data_search()
                st.session_state.data_search_results = results
                
            elif selected_step == "모델 학습만":
                status_text.text("🎯 모델 학습 중...")
                progress_bar.progress(50)
                
                results = system.step2_model_training(force_retrain)
                st.session_state.model_training_results = results
                
            elif selected_step == "예측만":
                status_text.text("📈 예측 실행 중...")
                progress_bar.progress(50)
                
                results = system.step3_prediction()
                st.session_state.prediction_results = results
                
            elif selected_step == "토론만":
                status_text.text("💬 토론 실행 중...")
                progress_bar.progress(50)
                
                results = system.step4_debate_rounds({}, debate_rounds)
                st.session_state.debate_results = results
            
            progress_bar.progress(100)
            status_text.text("✅ 분석 완료!")
            
        except Exception as e:
            st.error(f"❌ 분석 실패: {str(e)}")
            status_text.text("❌ 분석 실패")

# 메인 콘텐츠
st.title("🚀 하이브리드 주식 예측 시스템")
st.markdown("원래 capstone 구조를 기반으로 ML과 LLM을 통합한 새로운 시스템")

# 결과 표시
if hasattr(st.session_state, 'analysis_results'):
    results = st.session_state.analysis_results
    
    # 탭 생성
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 전체 결과",
        "🔍 데이터 수집",
        "🎯 모델 학습", 
        "📈 ML 예측",
        "💬 LLM 토론"
    ])
    
    with tab1:
        st.subheader("📊 전체 분석 결과")
        
        # 최종 예측 표시
        final_consensus = results['final_consensus']
        if final_consensus['final_prediction'] is not None:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "🎯 최종 예측",
                    f"${final_consensus['final_prediction']:.2f}",
                    help="ML과 LLM 분석을 종합한 최종 예측값"
                )
            
            with col2:
                ml_pred = final_consensus.get('ml_prediction')
                if ml_pred is not None:
                    st.metric(
                        "🤖 ML 예측",
                        f"${ml_pred:.2f}",
                        help="머신러닝 모델의 예측값"
                    )
            
            with col3:
                llm_pred = final_consensus.get('llm_prediction')
                if llm_pred is not None:
                    st.metric(
                        "🧠 LLM 예측",
                        f"${llm_pred:.2f}",
                        help="LLM 토론의 예측값"
                    )
            
            # 분석 근거
            st.markdown("### 📝 분석 근거")
            for reason in final_consensus['reasoning']:
                st.markdown(f"• {reason}")
        
        else:
            st.warning("⚠️ 최종 예측 결과가 없습니다.")
    
    with tab2:
        st.subheader("🔍 데이터 수집 결과")
        
        data_results = results['data_search']
        for agent_type, filepath in data_results.items():
            if filepath:
                st.success(f"✅ {agent_type.title()}: {filepath}")
            else:
                st.error(f"❌ {agent_type.title()}: 수집 실패")
    
    with tab3:
        st.subheader("🎯 모델 학습 결과")
        
        training_results = results['model_training']
        for agent_type, success in training_results.items():
            if success:
                st.success(f"✅ {agent_type.title()}: 학습 완료")
            else:
                st.error(f"❌ {agent_type.title()}: 학습 실패")
    
    with tab4:
        st.subheader("📈 ML 예측 결과")
        
        ml_results = results['ml_prediction']
        if ml_results['success']:
            st.success("✅ ML 예측 성공")
            
            # 예측값 표시
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "🎯 합의 예측",
                    f"${ml_results['consensus']:.2f}",
                    help="모든 에이전트의 가중평균"
                )
            
            with col2:
                st.metric(
                    "📊 에이전트 수",
                    len(ml_results['predictions']),
                    help="예측에 참여한 에이전트 수"
                )
            
            # 각 에이전트별 예측
            st.markdown("### 📊 에이전트별 예측")
            predictions_df = pd.DataFrame([
                {
                    'Agent': agent_type.title(),
                    'Prediction': f"${pred:.2f}",
                    'Weight': f"{ml_results['weights'][agent_type]:.1%}",
                    'Confidence': f"{ml_results['beta_values'][agent_type]:.3f}"
                }
                for agent_type, pred in ml_results['predictions'].items()
            ])
            
            st.dataframe(predictions_df, use_container_width=True)
            
            # 신뢰도 차트
            fig = px.bar(
                x=list(ml_results['beta_values'].keys()),
                y=list(ml_results['beta_values'].values()),
                title="에이전트별 신뢰도 (β)",
                labels={'x': 'Agent', 'y': 'Confidence (β)'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.error(f"❌ ML 예측 실패: {ml_results['error']}")
    
    with tab5:
        st.subheader("💬 LLM 토론 결과")
        
        debate_results = results['llm_debate']
        if debate_results['success']:
            st.success("✅ LLM 토론 성공")
            
            # 토론 로그 표시
            if debate_results['logs']:
                st.markdown("### 📝 토론 로그")
                for i, log in enumerate(debate_results['logs']):
                    with st.expander(f"라운드 {i+1}"):
                        st.json(log)
            
            # 최종 의견 표시
            if debate_results['final']:
                st.markdown("### 🎯 최종 의견")
                final = debate_results['final']
                st.json(final)
        
        else:
            st.error(f"❌ LLM 토론 실패: {debate_results['error']}")

# 개별 단계 결과 표시
elif hasattr(st.session_state, 'data_search_results'):
    st.subheader("🔍 데이터 수집 결과")
    results = st.session_state.data_search_results
    for agent_type, filepath in results.items():
        if filepath:
            st.success(f"✅ {agent_type.title()}: {filepath}")
        else:
            st.error(f"❌ {agent_type.title()}: 수집 실패")

elif hasattr(st.session_state, 'model_training_results'):
    st.subheader("🎯 모델 학습 결과")
    results = st.session_state.model_training_results
    for agent_type, success in results.items():
        if success:
            st.success(f"✅ {agent_type.title()}: 학습 완료")
        else:
            st.error(f"❌ {agent_type.title()}: 학습 실패")

elif hasattr(st.session_state, 'prediction_results'):
    st.subheader("📈 ML 예측 결과")
    results = st.session_state.prediction_results
    if results['success']:
        st.success("✅ ML 예측 성공")
        st.metric("🎯 합의 예측", f"${results['consensus']:.2f}")
    else:
        st.error(f"❌ ML 예측 실패: {results['error']}")

elif hasattr(st.session_state, 'debate_results'):
    st.subheader("💬 LLM 토론 결과")
    results = st.session_state.debate_results
    if results['success']:
        st.success("✅ LLM 토론 성공")
    else:
        st.error(f"❌ LLM 토론 실패: {results['error']}")

else:
    # 초기 화면
    st.markdown("""
    ## 🎯 하이브리드 주식 예측 시스템
    
    이 시스템은 **원래 capstone 구조**를 기반으로 **ML과 LLM을 통합**한 새로운 하이브리드 시스템입니다.
    
    ### 🔄 새로운 분석 흐름:
    
    #### 1️⃣ **TICKER 입력**
    - 사용자가 주식 티커 입력 (예: RZLV, AAPL, TSLA)
    
    #### 2️⃣ **각 Agent의 Searcher**
    - **목적**: 2022~2025년 CSV 파일 생성
    - **기능**: 
      - Fundamental Agent: 재무 데이터 수집
      - Technical Agent: 기술적 지표 데이터 수집  
      - Sentimental Agent: 감정 분석 데이터 수집
    
    #### 3️⃣ **각 Agent의 Trainer** (선택사항)
    - **목적**: 2022~2024년 데이터로 개별 Agent 학습
    - **기능**:
      - 이미 학습된 `.pt` 파일이 있으면 선택하여 로드
      - 없으면 새로 학습 실행
      - 실행하면 기존 모델 업데이트
    
    #### 4️⃣ **각 Agent의 Predicter**
    - **목적**: 상호학습 + 예측
    - **기능**:
      - 최근 1년 데이터로 상호학습 진행
      - 상호학습 후 최근 7일 데이터로 다음날 종가 예측
    
    #### 5️⃣ **Debate Round 진행**
    - **Reviewer Draft**: Opinion 생성
    - **Reviewer Rebut**: 반론/지지 의견 형성
    - **Reviewer Revise**: 예측 수정
    
    ### 🚀 사용 방법:
    1. 왼쪽 사이드바에서 주식 티커 입력
    2. 분석 단계 선택 (전체 분석 권장)
    3. "분석 시작" 버튼 클릭
    4. 결과를 탭별로 확인
    
    ### 🎯 지원 종목:
    - **기술주**: AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA
    - **금융주**: JPM, BAC, WFC, GS, MS
    - **에너지주**: XOM, CVX, COP, EOG
    - **기타**: RZLV, SPY, QQQ
    """)

# 푸터
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        🚀 하이브리드 주식 예측 시스템 | 원래 capstone 구조 기반 | ML + LLM 통합
    </div>
    """,
    unsafe_allow_html=True
)

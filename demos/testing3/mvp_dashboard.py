#!/usr/bin/env python3
"""
MVP 하이브리드 주식 예측 시스템 - Streamlit 대시보드
간단하고 깔끔한 UI
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import sys

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 시스템 import
from mvp_main import MVPHybridSystem

# 페이지 설정
st.set_page_config(
    page_title="🚀 MVP 하이브리드 주식 예측 시스템",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 사이드바
st.sidebar.title("🎯 MVP 하이브리드 주식 예측 시스템")
st.sidebar.markdown("---")

# 입력 설정
ticker = st.sidebar.text_input(
    "📈 주식 티커",
    value="RZLV",
    help="분석할 주식 티커를 입력하세요 (예: RZLV, AAPL, TSLA)"
).upper()

# 토론 라운드 수 (현재는 사용하지 않지만 향후 확장을 위해 유지)
debate_rounds = st.sidebar.slider(
    "💬 토론 라운드 수",
    min_value=1,
    max_value=5,
    value=3,
    help="LLM 토론에서 진행할 라운드 수 (현재는 해석만 사용)"
)

# 분석 실행 버튼
if st.sidebar.button("🚀 전체 파이프라인 실행", type="primary"):
    if not ticker:
        st.error("❌ 주식 티커를 입력해주세요.")
    else:
        # 진행 상황 표시
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # 시스템 초기화
            status_text.text("🔄 시스템 초기화 중...")
            progress_bar.progress(5)
            
            system = MVPHybridSystem(ticker)
            
            # 전체 파이프라인 자동 실행
            status_text.text("🔍 1단계: 데이터 수집 중...")
            progress_bar.progress(15)
            
            # 1단계: 데이터 수집
            data_results = system.step1_data_search()
            
            status_text.text("🎯 2단계: 모델 학습 중...")
            progress_bar.progress(35)
            
            # 2단계: 모델 학습 (기존 모델 있으면 로드, 없으면 학습)
            training_results = system.step2_model_training(force_retrain=False)
            
            status_text.text("📈 3단계: ML 예측 중...")
            progress_bar.progress(60)
            
            # 3단계: ML 예측
            ml_results = system.step3_prediction()
            
            status_text.text("💭 4단계: LLM 해석 중...")
            progress_bar.progress(85)
            
            # 4단계: LLM 해석
            if ml_results['success']:
                interpretation_results = system.step4_llm_interpretation(ml_results)
            else:
                interpretation_results = {'success': False, 'error': 'ML 예측 실패로 인한 해석 불가'}
            
            # 최종 결과 구성
            final_results = {
                'ticker': ticker,
                'data_search': data_results,
                'model_training': training_results,
                'ml_prediction': ml_results,
                'llm_interpretation': interpretation_results,
                'final_result': {
                    'prediction': ml_results.get('consensus', 0.0),
                    'interpretation': interpretation_results.get('interpretation', '해석 없음'),
                    'confidence': 'medium'
                },
                'timestamp': None
            }
            
            # 결과를 세션 상태에 저장
            st.session_state.mvp_results = final_results
            
            progress_bar.progress(100)
            status_text.text("✅ 전체 파이프라인 완료!")
            
        except Exception as e:
            st.error(f"❌ 분석 실패: {str(e)}")
            status_text.text("❌ 분석 실패")

# 메인 콘텐츠
st.title("🚀 MVP 하이브리드 주식 예측 시스템")
st.markdown("**ML 예측 + LLM 해석**의 간단하고 효율적인 시스템")

# 결과 표시
if hasattr(st.session_state, 'mvp_results'):
    results = st.session_state.mvp_results
    
    # 탭 생성
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 전체 결과",
        "🔍 데이터 수집",
        "🎯 모델 학습", 
        "📈 ML 예측",
        "💭 LLM 해석"
    ])
    
    with tab1:
        st.subheader("📊 전체 분석 결과")
        
        # 최종 예측 표시
        final_result = results['final_result']
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "🎯 최종 예측",
                f"${final_result['prediction']:.2f}",
                help="ML 모델들의 가중평균 예측값"
            )
        
        with col2:
            st.metric(
                "📊 에이전트 수",
                len(results['ml_prediction']['predictions']),
                help="예측에 참여한 에이전트 수"
            )
        
        with col3:
            st.metric(
                "🎯 신뢰도",
                final_result['confidence'],
                help="예측의 신뢰도"
            )
        
        # LLM 해석 표시
        st.markdown("### 💭 LLM 해석")
        st.text_area(
            "분석 결과",
            final_result['interpretation'],
            height=200,
            disabled=True
        )
    
    with tab2:
        st.subheader("🔍 데이터 수집 결과")
        
        data_results = results['data_search']
        for agent_type, filepath in data_results.items():
            if filepath:
                st.success(f"✅ {agent_type}: {filepath}")
            else:
                st.error(f"❌ {agent_type}: 수집 실패")
    
    with tab3:
        st.subheader("🎯 모델 학습 결과")
        
        training_results = results['model_training']
        for agent_type, success in training_results.items():
            if success:
                st.success(f"✅ {agent_type}: 학습 완료")
            else:
                st.error(f"❌ {agent_type}: 학습 실패")
    
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
                    'Agent': agent_type.replace('Agent', ''),
                    'Prediction': f"${pred['prediction']:.2f}",
                    'Uncertainty': f"{pred['uncertainty']:.4f}",
                    'Confidence (β)': f"{pred['beta']:.3f}"
                }
                for agent_type, pred in ml_results['predictions'].items()
            ])
            
            st.dataframe(predictions_df, use_container_width=True)
            
            # 신뢰도 차트
            fig = px.bar(
                x=[agent.replace('Agent', '') for agent in ml_results['predictions'].keys()],
                y=[pred['beta'] for pred in ml_results['predictions'].values()],
                title="에이전트별 신뢰도 (β)",
                labels={'x': 'Agent', 'y': 'Confidence (β)'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.error(f"❌ ML 예측 실패")
    
    with tab5:
        st.subheader("💭 LLM 해석 결과")
        
        interpretation_results = results['llm_interpretation']
        if interpretation_results['success']:
            st.success("✅ LLM 해석 성공")
            
            # 해석 내용 표시
            st.markdown("### 📝 해석 내용")
            st.text_area(
                "분석 결과",
                interpretation_results['interpretation'],
                height=300,
                disabled=True
            )
        
        else:
            st.error(f"❌ LLM 해석 실패: {interpretation_results['error']}")

else:
    # 초기 화면
    st.markdown("""
    ## 🎯 MVP 하이브리드 주식 예측 시스템
    
    이 시스템은 **ML 예측 + LLM 해석**의 간단하고 효율적인 MVP 모델입니다.
    
    ### 🔄 자동 파이프라인:
    
    #### 1️⃣ **TICKER 입력**
    - 왼쪽 사이드바에서 주식 티커 입력 (예: RZLV, AAPL, TSLA)
    
    #### 2️⃣ **자동 실행**
    - "🚀 전체 파이프라인 실행" 버튼 클릭
    - 모든 단계가 자동으로 순차 실행됩니다:
      - 🔍 **데이터 수집**: 각 Agent별로 2022~2025년 데이터 수집
      - 🎯 **모델 학습**: 기존 모델 로드 또는 새로 학습
      - 📈 **ML 예측**: 각 Agent별 예측 + 가중평균 합의
      - 💭 **LLM 해석**: 예측 결과에 대한 투자 의견 제공
    
    ### 🚀 사용 방법:
    1. 왼쪽 사이드바에서 주식 티커 입력
    2. "🚀 전체 파이프라인 실행" 버튼 클릭
    3. 자동으로 모든 단계가 실행됩니다
    4. 결과를 탭별로 확인
    
    ### 🎯 핵심 특징:
    - **간단함**: 복잡한 토론 제거, 핵심 기능만 유지
    - **자동화**: 한 번의 클릭으로 전체 파이프라인 실행
    - **효율성**: ML 예측 + LLM 해석의 최적 조합
    - **투명성**: 각 단계별 결과를 명확히 표시
    - **실용성**: 실제 투자 결정에 도움이 되는 정보 제공
    
    ### 📊 지원 종목:
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
        🚀 MVP 하이브리드 주식 예측 시스템 | ML 예측 + LLM 해석 | 간단하고 효율적
    </div>
    """,
    unsafe_allow_html=True
)

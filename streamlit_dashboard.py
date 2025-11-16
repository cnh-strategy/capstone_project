import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import time
from agents.debate_agent import DebateAgent

# 페이지 설정
st.set_page_config(
    page_title="Stock Debate Dashboard",
    page_icon="■",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일링
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# 제목
st.markdown('<h1 class="main-header">■ Stock Debate Dashboard</h1>', unsafe_allow_html=True)

# 사이드바 설정
st.sidebar.header("※ 설정")

# 종목 입력
ticker = st.sidebar.text_input(
    "종목 티커 입력", 
    value="AAPL",
    help="예: AAPL, TSLA, MSFT, GOOGL, NVDA, AMZN, META, NFLX, BABA, PLTR, AMD, INTC, CRM, ADBE"
).upper()

rounds = st.sidebar.slider("라운드 수", 1, 5, 3)

# 티커 유효성 검사
if not ticker or len(ticker) < 1:
    st.sidebar.error("종목 티커를 입력해주세요.")
elif len(ticker) > 10:
    st.sidebar.error("티커는 10자 이하로 입력해주세요.")
else:
    st.sidebar.success(f"선택된 종목: **{ticker}**")

# 추가 옵션
st.sidebar.markdown("### ※ 디스플레이 옵션")
show_annotations = st.sidebar.checkbox("차트에 값 표시", value=True)
chart_height = st.sidebar.slider("차트 높이", 300, 800, 400)

# 데이터 새로고침 버튼
if st.sidebar.button("※ 데이터 새로고침"):
    st.cache_data.clear()
    st.session_state.real_debate_data = None  # 실제 토론 데이터도 클리어
    st.rerun()

# 목 데이터 함수 제거됨 - 실제 토론만 사용

# 세션 상태 초기화
if 'real_debate_data' not in st.session_state:
    st.session_state.real_debate_data = None

# debate_ver3의 DebateAgent를 사용한 토론 실행 함수
def run_debate_with_detailed_progress(debate_agent, ticker, rounds, status_text, progress_bar):
    """debate_ver3의 DebateAgent를 사용하여 토론 실행"""
    import time
    import statistics
    
    # 초기 설정
    status_text.text("■ 토론 시스템 초기화 중...")
    progress_bar.progress(10)
    time.sleep(0.5)
    
    # Round 0: 초기 의견 생성
    status_text.text("■ Round 0 - 초기 의견 생성 중...")
    progress_bar.progress(20)
    debate_agent.get_opinion(0, ticker)
    time.sleep(0.5)
    
    # 각 라운드별 토론 수행
    for r in range(1, rounds + 1):
        status_text.text(f"■ Round {r} - 반박/지지 생성 중...")
        progress_bar.progress(20 + (r - 1) * (60 // rounds))
        
        # 반박/지지 생성
        debate_agent.get_rebuttal(r)
        time.sleep(0.3)
        
        status_text.text(f"■ Round {r} - 의견 수정 중...")
        progress_bar.progress(20 + r * (60 // rounds))
        
        # 의견 수정
        debate_agent.get_revise(r)
        time.sleep(0.3)
        
        status_text.text(f"■ Round {r} 완료!")
        time.sleep(0.5)
    
    # 최종 결과 계산
    status_text.text("■ 최종 결과 계산 중...")
    progress_bar.progress(90)
    
    # DebateAgent의 get_ensemble 메서드 사용
    ensemble = debate_agent.get_ensemble()
    
    return debate_agent, ensemble

# debate_ver3의 DebateAgent를 사용한 토론 실행 함수
def run_real_debate_with_progress(ticker, rounds):
    """debate_ver3의 DebateAgent를 사용하여 토론 실행"""
    try:
        # 진행 상황 표시를 위한 컨테이너
        progress_container = st.container()
        status_container = st.container()
        
        with progress_container:
            st.subheader("■ 토론 진행 상황")
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        # DebateAgent 초기화
        status_text.text("■ DebateAgent 초기화 중...")
        progress_bar.progress(10)
        
        debate_agent = DebateAgent(rounds=rounds, ticker=ticker)
        status_text.text("■ 에이전트 생성 중...")
        progress_bar.progress(20)
        
        # 에이전트 정보 표시
        with status_container:
            st.info("**참여 에이전트:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("■ **MacroAgent**")
                st.write("거시경제 센티멘탈 분석가")
                st.write("예측 범위: ±12%")
            with col2:
                st.write("■ **SentimentalAgent**")
                st.write("중립적 센티멘탈 분석가")
                st.write("예측 범위: ±10%")
            with col3:
                st.write("■ **TechnicalAgent**")
                st.write("공격적 기술적 분석가")
                st.write("예측 범위: ±15%")
        
        # 토론 실행 (진행 상황과 함께)
        status_text.text("■ 실제 토론 시작... (각 라운드마다 LLM API 호출로 시간이 오래 걸립니다)")
        progress_bar.progress(30)
        
        # 실제 토론 실행 - 세부 단계별 진행상황 표시
        debate_agent, final = run_debate_with_detailed_progress(debate_agent, ticker, rounds, status_text, progress_bar)
        
        # 실제 토론 완료 확인
        st.write(f"■ {ticker} 실제 토론 완료!")
        st.write(f"■ 현재가: {final.get('last_price', 'None')} {final.get('currency', 'USD')}")
        
        # 세션 상태에 실제 토론 결과 저장
        st.session_state.real_debate_data = {
            'debate_agent': debate_agent,
            'final': final,
            'ticker': ticker,
            'rounds': rounds
        }
        
        # 진행 상황 업데이트
        status_text.text("■ 토론 완료! 결과 정리 중...")
        progress_bar.progress(90)
        
        status_text.text("■ 토론 완료!")
        progress_bar.progress(100)
        
        # 진행 상황 컨테이너 숨기기
        time.sleep(1)
        progress_container.empty()
        status_container.empty()
        
        return debate_agent, final
        
    except Exception as e:
        st.error(f"토론 실행 중 오류 발생: {e}")
        import traceback
        st.error(f"상세 오류: {traceback.format_exc()}")
        # 오류 발생 시 빈 데이터 반환
        return None, {}


# 토론 시작 버튼
if st.sidebar.button("■ 토론 시작", type="primary"):
    # 진행 상황과 함께 토론 실행
    debate_agent, final = run_real_debate_with_progress(ticker, rounds)
    
    # 완료 메시지
    st.success("■ 토론이 성공적으로 완료되었습니다!")
    
    # 결과 요약 표시
    with st.expander("■ 토론 결과 요약", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("■ 분석 종목", ticker)
            st.metric("■ 실행 라운드", rounds)
            st.metric("■ 참여 에이전트", "3명")
        with col2:
            st.metric("■ 최종 평균 예측", f"{final.get('mean_next_close', 0):.2f}")
            st.metric("■ 중앙값", f"{final.get('median_next_close', 0):.2f}")
            st.metric("■ 표준편차", f"{final.get('std_next_close', 0):.2f}")
        with col3:
            current_price = final.get('last_price') or final.get('current_price') or 0
            st.metric("■ 현재가", f"{current_price:.2f}" if current_price else "N/A")
            
            # 예측 범위 계산
            agent_prices = [final.get(f'{agent}_next_close', 0) for agent in ['SentimentalAgent', 'TechnicalAgent', 'MacroAgent']]
            if any(agent_prices):
                min_price = min(agent_prices)
                max_price = max(agent_prices)
                st.metric("■ 예측 범위", f"{min_price:.2f} ~ {max_price:.2f}")
                
                # 변동성 계산
                if current_price and current_price > 0:
                    volatility = ((max_price - min_price) / current_price * 100)
                    st.metric("■ 변동성", f"{volatility:.1f}%")
                else:
                    st.metric("■ 변동성", "N/A")
            else:
                st.metric("■ 예측 범위", "N/A")
                st.metric("■ 변동성", "N/A")
    
    # 라운드별 상세 결과 표시
    with st.expander("■ 라운드별 상세 결과", expanded=False):
        for round_no in range(rounds + 1):
            if round_no in debate_agent.opinions:
                st.subheader(f"■ 라운드 {round_no}")
                
                # 각 에이전트의 의견 표시
                for agent_id, opinion in debate_agent.opinions[round_no].items():
                    agent_name = agent_id.replace('Agent', '')
                    price = opinion.target.next_close
                    reason = getattr(opinion, 'reason', '') or getattr(opinion, 'reasoning', '분석 결과')
                    
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.write(f"**{agent_name}:** {price:.2f}")
                    with col2:
                        st.write(f"*{reason[:100]}{'...' if len(reason) > 100 else ''}*")
                
                # 반박/지지 결과 표시
                if round_no in debate_agent.rebuttals and debate_agent.rebuttals[round_no]:
                    st.write("**■ 반박/지지 결과:**")
                    for rebuttal in debate_agent.rebuttals[round_no]:
                        from_agent = rebuttal.from_agent_id.replace('Agent', '')
                        to_agent = rebuttal.to_agent_id.replace('Agent', '')
                        stance = "■ 지지" if rebuttal.stance == "SUPPORT" else "■ 반박"
                        message = getattr(rebuttal, 'message', '')
                        st.write(f"- {from_agent} → {to_agent}: {stance}")
                        if message:
                            st.write(f"  *{message[:100]}{'...' if len(message) > 100 else ''}*")
                
                st.markdown("---")
    
    st.rerun()
else:
    # 실제 토론 데이터가 있으면 사용, 없으면 안내 메시지
    if st.session_state.real_debate_data:
        debate_agent = st.session_state.real_debate_data['debate_agent']
        final = st.session_state.real_debate_data['final']
        st.success("■ 실제 토론 데이터를 사용 중입니다.")
    else:
        # 초기 상태
        debate_agent, final = None, {}

# 종목 정보 섹션 (티커가 입력된 경우에만 표시)
if ticker and len(ticker) >= 1:
    st.markdown("---")
    
    try:
        import yfinance as yf
        
        # 종목 정보 가져오기
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # 최근 주가 차트 (7일) - 먼저 표시
        try:
            hist = stock.history(period="7d")
            if not hist.empty:
                st.subheader("■ 최근 7일 주가 차트")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=hist.index,
                    y=hist['Close'],
                    mode='lines+markers',
                    name='종가',
                    line=dict(color='#1f77b4', width=2),
                    marker=dict(size=6)
                ))
                
                fig.update_layout(
                    title=f"{ticker} 7일 주가 추이",
                    xaxis_title="날짜",
                    yaxis_title="가격 ($)",
                    height=400,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"※ 주가 차트를 불러올 수 없습니다: {str(e)}")
        
        # 기본 정보 컬럼 - 차트 밑에 배치
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
            st.metric("■ 현재가", f"${current_price:.2f}")
        
        with col2:
            market_cap = info.get('marketCap', 0)
            if market_cap > 1e12:
                market_cap_display = f"${market_cap/1e12:.2f}T"
            elif market_cap > 1e9:
                market_cap_display = f"${market_cap/1e9:.2f}B"
            elif market_cap > 1e6:
                market_cap_display = f"${market_cap/1e6:.2f}M"
            else:
                market_cap_display = f"${market_cap:.0f}"
            st.metric("■ 시가총액", market_cap_display)
        
        with col3:
            pe_ratio = info.get('trailingPE', 0)
            st.metric("■ PER", f"{pe_ratio:.2f}" if pe_ratio else "N/A")
        
        with col4:
            volume = info.get('volume', 0)
            if volume > 1e9:
                volume_display = f"{volume/1e9:.2f}B"
            elif volume > 1e6:
                volume_display = f"{volume/1e6:.2f}M"
            else:
                volume_display = f"{volume:,.0f}"
            st.metric("■ 거래량", volume_display)
        
        # 추가 정보
        col5, col6 = st.columns(2)
        
        with col5:
            st.write(f"**■ 회사명:** {info.get('longName', 'N/A')}")
            st.write(f"**■ 섹터:** {info.get('sector', 'N/A')}")
            st.write(f"**■ 업종:** {info.get('industry', 'N/A')}")
        
        with col6:
            st.write(f"**■ 52주 최고가:** ${info.get('fiftyTwoWeekHigh', 0):.2f}")
            st.write(f"**■ 52주 최저가:** ${info.get('fiftyTwoWeekLow', 0):.2f}")
            st.write(f"**■ 배당수익률:** {info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else "**■ 배당수익률:** N/A")
        
        # 구분선 추가
        st.markdown("---")
            
    except Exception as e:
        st.error(f"※ {ticker} 종목 정보를 불러올 수 없습니다: {str(e)}")

# 요약 메트릭
col1, col2, col3, col4 = st.columns(4)
with col1:
    mean_price = final.get('mean_next_close', 0)
    st.metric("■ 평균 예측가격", f"{mean_price:.2f}" if mean_price else "N/A")
with col2:
    median_price = final.get('median_next_close', 0)
    st.metric("■ 중앙값", f"{median_price:.2f}" if median_price else "N/A")
with col3:
    current_price = final.get('last_price') or final.get('current_price') or 0
    st.metric("■ 현재가", f"{current_price:.2f}" if current_price else "N/A")
with col4:
    st.metric("■ 총 라운드", rounds if debate_agent else 0)

# 데이터 소스 표시
if st.session_state.real_debate_data:
    st.success("※ 실제 토론 데이터를 사용 중입니다")
else:
    st.info("※ '토론 시작' 버튼을 눌러 실제 토론을 실행하세요.")

st.markdown("---")

# 탭으로 구분
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "■ 최종의견 표", 
    "■ 투자의견 표", 
    "■ 최종 예측 비교", 
    "■ 라운드별 의견 변화", 
    "■ 반박/지지 패턴"
])

# 최종의견 표
with tab1:
    st.header("■ 최종의견 표")
    
    # 최종 의견 데이터 준비
    final_opinions_data = []
    if debate_agent and debate_agent.opinions:
        # 최종 라운드의 의견 가져오기
        final_round = max(debate_agent.opinions.keys())
        final_opinions = debate_agent.opinions[final_round]
        
        for agent_id, opinion in final_opinions.items():
            display_name = agent_id.replace('Agent', '')
            price = opinion.target.next_close
            reasoning_text = getattr(opinion, 'reasoning', None) or getattr(opinion, 'reason', '')
            total_opinion = reasoning_text[:100] + "..." if len(reasoning_text) > 100 else reasoning_text
            
            final_opinions_data.append({
                '에이전트': display_name,
                '최종 예측 가격': f"{price:.2f}",
                '전체 투자의견': total_opinion
            })
    
    if final_opinions_data:
        df_final = pd.DataFrame(final_opinions_data)
        st.dataframe(df_final, use_container_width=True, hide_index=True)
    else:
        st.info("토론 데이터가 없습니다. '토론 시작' 버튼을 눌러주세요.")

# 투자의견 표
with tab2:
    st.header("■ 투자의견 표")
    
    # 투자의견 데이터 준비
    if debate_agent and debate_agent.opinions:
        st.markdown("### ■ 라운드별 에이전트 의견 상세")
        
        # 각 라운드별로 상세 표시
        for round_no in sorted(debate_agent.opinions.keys()):
            with st.expander(f"■ Round {round_no} - 에이전트 의견", expanded=(round_no == 0)):
                st.markdown(f"**라운드 {round_no}에서의 각 에이전트 의견:**")
                
                # 에이전트별 의견 표시
                for agent_id, opinion in debate_agent.opinions[round_no].items():
                    agent_name = agent_id.replace('Agent', '')
                    price = opinion.target.next_close
                    reason = getattr(opinion, 'reason', '') or getattr(opinion, 'reasoning', '')
                    
                    # 에이전트별 색상 구분
                    if 'Sentimental' in agent_name:
                        color = "■"
                    elif 'Technical' in agent_name:
                        color = "■"
                    else:
                        color = "■"
                    
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.metric(f"{color} **{agent_name}**", f"{price:.2f}")
                    with col2:
                        if reason:
                            st.write(f"**의견:** {reason}")
                        else:
                            st.write("*의견 없음*")
                
                st.markdown("---")
        
        # 요약 테이블
        st.markdown("### 📋 요약 테이블")
        opinions_data = []
        agent_order = ['SentimentalAgent', 'TechnicalAgent', 'MacroAgent']
        
        for round_no in sorted(debate_agent.opinions.keys()):
            row = {'라운드': f"Round {round_no}"}
            for agent_id in agent_order:
                if agent_id in debate_agent.opinions[round_no]:
                    opinion = debate_agent.opinions[round_no][agent_id]
                    price = opinion.target.next_close
                    reasoning_text = getattr(opinion, 'reasoning', None) or getattr(opinion, 'reason', '')
                    if reasoning_text:
                        reasoning = reasoning_text[:80] + "..." if len(reasoning_text) > 80 else reasoning_text
                        row[agent_id.replace('Agent', '')] = f"**{price:.2f}**\n\n*{reasoning}*"
                    else:
                        row[agent_id.replace('Agent', '')] = f"**{price:.2f}**"
                else:
                    row[agent_id.replace('Agent', '')] = "-"
            opinions_data.append(row)
        
        if opinions_data:
            df_opinions = pd.DataFrame(opinions_data)
            st.dataframe(df_opinions, use_container_width=True, hide_index=True, height=400)
    else:
        st.info("투자의견 데이터가 없습니다. '토론 시작' 버튼을 눌러주세요.")

# 최종 예측 비교
with tab3:
    st.header("■ 최종 예측 비교")
    
    if debate_agent and debate_agent.opinions:
        final_round = max(debate_agent.opinions.keys())
        final_opinions = debate_agent.opinions[final_round]
        
        final_agents = []
        final_prices = []
        for agent_id, opinion in final_opinions.items():
            agent_name = agent_id.replace('Agent', '')
            final_agents.append(agent_name)
            final_prices.append(opinion.target.next_close)
        
        if final_agents and final_prices:
            # 막대차트
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=final_agents,
                y=final_prices,
                marker_color=['lightblue', 'lightcoral', 'lightgreen'],
                text=[f"{price:.2f}" for price in final_prices],
                textposition='auto'
            ))
            
            # 평균선 추가
            mean_price = final.get('mean_next_close', np.mean(final_prices))
            fig.add_hline(y=mean_price, line_dash="dot", line_color="red", 
                         annotation_text=f"평균: {mean_price:.2f}")
            
            fig.update_layout(
                title="최종 예측 가격 비교",
                xaxis_title="에이전트",
                yaxis_title="예측 가격",
                height=chart_height
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("토론 데이터가 없습니다. '토론 시작' 버튼을 눌러주세요.")

# 라운드별 의견 변화
with tab4:
    st.header("■ 라운드별 의견 변화")
    
    if debate_agent and debate_agent.opinions:
        rounds_data = []
        agents_data = {}
        current_price = final.get('last_price') or final.get('current_price') or 0
        
        for round_no in sorted(debate_agent.opinions.keys()):
            rounds_data.append(round_no)
            for agent_id, opinion in debate_agent.opinions[round_no].items():
                if agent_id not in agents_data:
                    agents_data[agent_id] = []
                agents_data[agent_id].append(opinion.target.next_close)
        
        if agents_data:
            fig = go.Figure()
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            
            for i, (agent_id, prices) in enumerate(agents_data.items()):
                display_name = agent_id.replace('Agent', '')
                fig.add_trace(go.Scatter(
                    x=rounds_data,
                    y=prices,
                    mode='lines+markers',
                    name=display_name,
                    line=dict(color=colors[i % len(colors)], width=3),
                    marker=dict(size=8),
                    text=[f"{price:.2f}" for price in prices],
                    textposition="top center"
                ))
            
            # 현재가 라인 추가
            if current_price and current_price > 0:
                fig.add_hline(y=current_price, line_dash="dash", line_color="black",
                             annotation_text=f"현재가: {current_price:.2f}")
            
            fig.update_layout(
                title="라운드별 에이전트 의견 변화",
                xaxis_title="라운드",
                yaxis_title="예측 가격",
                height=chart_height
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("토론 데이터가 없습니다. '토론 시작' 버튼을 눌러주세요.")

# 반박/지지 패턴
with tab5:
    st.header("■ 반박/지지 패턴")
    
    # 반박/지지 데이터 준비
    if debate_agent and debate_agent.rebuttals:
        agent_rebuttal_data = {}
        all_rebuttals = []  # 모든 반박/지지 메시지 저장
        
        for round_no, rebuttals in debate_agent.rebuttals.items():
            for rebuttal in rebuttals:
                from_agent = rebuttal.from_agent_id.replace('Agent', '')
                to_agent = rebuttal.to_agent_id.replace('Agent', '')
                
                if from_agent not in agent_rebuttal_data:
                    agent_rebuttal_data[from_agent] = {'REBUT': 0, 'SUPPORT': 0}
                agent_rebuttal_data[from_agent][rebuttal.stance] += 1
                
                # 상세 정보 저장
                all_rebuttals.append({
                    'round': round_no,
                    'from_agent': from_agent,
                    'to_agent': to_agent,
                    'stance': rebuttal.stance,
                    'message': getattr(rebuttal, 'message', '')
                })
        
        if agent_rebuttal_data:
            # 상세 반박/지지 내역 표시
            st.markdown("### ■ 반박/지지 상세 내역")
            
            if all_rebuttals:
                for i, rebuttal in enumerate(all_rebuttals):
                    with st.expander(f"■ Round {rebuttal['round']} - {rebuttal['from_agent']} → {rebuttal['to_agent']}", expanded=(i == 0)):
                        stance_emoji = "■ 반박" if rebuttal['stance'] == 'REBUT' else "■ 지지"
                        st.markdown(f"**{stance_emoji}**")
                        st.write(f"**발신자:** {rebuttal['from_agent']}")
                        st.write(f"**수신자:** {rebuttal['to_agent']}")
                        st.write(f"**라운드:** {rebuttal['round']}")
                        if rebuttal['message']:
                            st.write(f"**메시지:** {rebuttal['message']}")
                        else:
                            st.write("*메시지 없음*")
            else:
                st.info("반박/지지 데이터가 없습니다.")
            
            # 차트 표시
            st.markdown("### ■ 반박/지지 패턴 차트")
            agents = list(agent_rebuttal_data.keys())
            rebut_counts = [agent_rebuttal_data[agent]['REBUT'] for agent in agents]
            support_counts = [agent_rebuttal_data[agent]['SUPPORT'] for agent in agents]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=agents,
                y=rebut_counts,
                name='반박',
                marker_color='#FF6B6B',
                text=rebut_counts,
                textposition='auto'
            ))
            fig.add_trace(go.Bar(
                x=agents,
                y=support_counts,
                name='지지',
                marker_color='#4ECDC4',
                text=support_counts,
                textposition='auto'
            ))
            
            fig.update_layout(
                title="에이전트별 반박/지지 패턴",
                xaxis_title="에이전트",
                yaxis_title="개수",
                barmode='group',
                height=chart_height
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 요약 통계
            st.markdown("### ■ 반박/지지 요약 통계")
            col1, col2, col3 = st.columns(3)
            
            total_rebuts = sum(rebut_counts)
            total_supports = sum(support_counts)
            total_interactions = total_rebuts + total_supports
            
            with col1:
                st.metric("■ 총 반박 수", total_rebuts)
            with col2:
                st.metric("■ 총 지지 수", total_supports)
            with col3:
                st.metric("■ 총 상호작용", total_interactions)
            
            if total_interactions > 0:
                st.metric("■ 반박 비율", f"{(total_rebuts / total_interactions * 100):.1f}%")
        else:
            st.info("반박/지지 데이터가 없습니다.")
    else:
        st.info("토론 데이터가 없습니다. '토론 시작' 버튼을 눌러주세요.")

# 푸터
st.markdown("---")

# 정보 섹션
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### ■ 데이터 정보")
    data_points = len(debate_agent.opinions) * 3 if debate_agent and debate_agent.opinions else 0
    st.info(f"""
    - **종목**: {ticker}
    - **라운드 수**: {rounds}
    - **에이전트 수**: 3개
    - **데이터 포인트**: {data_points}개
    """)

with col2:
    st.markdown("### ■ 주요 지표")
    
    # 에이전트별 예측가 추출
    agent_prices = [final.get(f'{agent}_next_close', 0) for agent in ['SentimentalAgent', 'TechnicalAgent', 'MacroSentiAgent']]
    agent_prices = [p for p in agent_prices if p > 0]  # 0보다 큰 값만
    
    if agent_prices:
        min_price = min(agent_prices)
        max_price = max(agent_prices)
        std_dev = np.std(agent_prices)
        
        st.success(f"""
        - **예측 범위**: {min_price:.2f} ~ {max_price:.2f}
        - **표준편차**: {std_dev:.2f}
        - **변동성**: {'높음' if std_dev > 5 else '낮음'}
        """)
    else:
        st.success("""
        - **예측 범위**: N/A
        - **표준편차**: N/A
        - **변동성**: N/A
        """)

with col3:
    st.markdown("### ■ 사용 팁")
    st.warning("""
    - 사이드바에서 종목과 라운드 수 변경
    - 차트 높이와 표시 옵션 조정
    - 데이터 새로고침 버튼으로 실시간 업데이트
    - 탭을 클릭하여 각 섹션 탐색
    """)

st.markdown("---")
st.markdown("■ **Streamlit 대시보드** | ■ **실시간 데이터 시각화** | ■ **인터랙티브 차트**")

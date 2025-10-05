import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
# 목 데이터 import 제거됨

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

# 세부 진행상황과 함께 토론 실행하는 함수
def run_debate_with_detailed_progress(debate, ticker, rounds, status_text, progress_bar):
    """토론의 세부 단계별 진행상황을 표시하며 실행"""
    import time
    import statistics
    
    # 0단계: 각 에이전트 초안 생성
    status_text.text("■ 초기 의견 생성 중...")
    progress_bar.progress(35)
    
    for i, (aid, agent) in enumerate(debate.agents.items()):
        agent_name = aid.replace("Agent", "")
        status_text.text(f"Round 0 - [{agent_name}] Opinion 생성 중...")
        op = agent.reviewer_draft(ticker)
        progress_bar.progress(35 + (i + 1) * 5)  # 35, 40, 45
        time.sleep(0.5)  # 진행상황이 보이도록 잠시 대기
    
    # 공용 스냅샷 선택
    status_text.text("■ 공용 데이터 스냅샷 선택 중...")
    progress_bar.progress(50)
    common_sd = debate._choose_common_snapshot()
    
    # 라운드 수행
    for r in range(1, rounds + 1):
        status_text.text(f"■ Round {r} 시작...")
        progress_bar.progress(50 + (r - 1) * (40 // rounds))
        
        # 최신 의견 스냅샷
        latest = debate._latest_opinions()
        
        # 1) 반박/지지 생성
        status_text.text(f"Round {r} - 반박/지지 분석 중...")
        all_rebuttals = []
        
        for i, (my_id, agent) in enumerate(debate.agents.items()):
            agent_name = my_id.replace("Agent", "")
            status_text.text(f"Round {r} - [{agent_name}] 반박/지지 생성 중...")
            
            my_latest = latest.get(my_id)
            if not my_latest:
                continue
            others_latest = {oid: op for oid, op in latest.items() if oid != my_id}
            
            # 각 에이전트가 타인에 대한 Rebuttal을 생성
            rbts = agent.reviewer_rebut(
                round_num=r,
                my_lastest=my_latest,
                others_latest=others_latest,
                stock_data=common_sd,
            )
            all_rebuttals.extend(rbts or [])
            time.sleep(0.3)
        
        # 2) 반박/지지 반영하여 수정
        status_text.text(f"Round {r} - 의견 수정 중...")
        
        # 타겟 에이전트별로 받은 rebuttal 묶어주기
        received_by_agent = {aid: [] for aid in debate.agents.keys()}
        for rb in all_rebuttals:
            if rb.to_agent_id in received_by_agent:
                received_by_agent[rb.to_agent_id].append(rb)
        
        # 수정 실행
        for i, (my_id, agent) in enumerate(debate.agents.items()):
            agent_name = my_id.replace("Agent", "")
            status_text.text(f"Round {r} - [{agent_name}] 의견 수정 중...")
            
            my_latest = latest.get(my_id)
            if not my_latest:
                continue
            others_latest = {oid: op for oid, op in latest.items() if oid != my_id}
            recv = received_by_agent.get(my_id, [])
            
            revised = agent.reviewer_revise(
                my_lastest=my_latest,
                others_latest=others_latest,
                received_rebuttals=recv,
                stock_data=common_sd,
            )
            time.sleep(0.3)
        
        # 라운드 로그 적재
        from agents.base_agent import RoundLog
        round_log = RoundLog(
            round_no=r,
            opinions=[debate.agents[aid].opinions[-1] for aid in debate.agents.keys() if debate.agents[aid].opinions],
            rebuttals=all_rebuttals,
            summary=debate._summarize(),
        )
        debate.logs.append(round_log)
        
        status_text.text(f"■ Round {r} 완료!")
        time.sleep(0.5)
    
    # 최종 집계
    status_text.text("■ 최종 결과 계산 중...")
    progress_bar.progress(90)
    
    latest = debate._latest_opinions()
    final_points = [float(op.target.next_close) for op in latest.values() if op and op.target]
    ensemble = {
        "ticker": ticker,
        "agents": {aid: float(op.target.next_close) for aid, op in latest.items()},
        "mean_next_close": (statistics.fmean(final_points) if final_points else None),
        "median_next_close": (statistics.median(final_points) if final_points else None),
        "currency": (common_sd.currency if common_sd else None),
        "last_price": (float(common_sd.last_price) if (common_sd and common_sd.last_price is not None) else None),
    }
    
    return debate.logs, ensemble

# 실제 토론 실행 함수 (캐시 비활성화)
def run_real_debate_with_progress(ticker, rounds):
    """실제 토론 시스템을 실행하여 데이터 생성 (진행 상황 표시)"""
    try:
        from main import DebateSystem
        
        # 진행 상황 표시를 위한 컨테이너
        progress_container = st.container()
        status_container = st.container()
        
        with progress_container:
            st.subheader("🚀 토론 진행 상황")
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        # 토론 시스템 초기화
        status_text.text("📋 토론 시스템 초기화 중...")
        progress_bar.progress(10)
        
        debate_system = DebateSystem()
        status_text.text("👥 에이전트 생성 중...")
        progress_bar.progress(20)
        
        # 에이전트 정보 표시
        with status_container:
            st.info("**참여 에이전트:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("■ **FundamentalAgent**")
                st.write("보수적 펀더멘털 분석가")
                st.write("예측 범위: ±5%")
            with col2:
                st.write("■ **SentimentalAgent**")
                st.write("중립적 센티멘탈 분석가")
                st.write("예측 범위: ±10%")
            with col3:
                st.write("■ **TechnicalAgent**")
                st.write("공격적 기술적 분석가")
                st.write("예측 범위: ±15%")
        
        # 토론 실행 (라운드별 진행 상황 표시)
        status_text.text("■ 토론 시작...")
        progress_bar.progress(30)
        
        # Debate 클래스의 run 메서드를 직접 호출하여 진행 상황 추적
        agents = debate_system.create_agents()
        from debate_agent import Debate
        debate = Debate(agents, verbose=False)  # Streamlit에서는 verbose=False
        
        # 실제 토론 실행 (진행 상황과 함께)
        status_text.text("■ 실제 토론 시작... (각 라운드마다 LLM API 호출로 시간이 오래 걸립니다)")
        progress_bar.progress(30)
        
        # 실제 토론 실행 - 세부 단계별 진행상황 표시
        logs, final = run_debate_with_detailed_progress(debate, ticker, rounds, status_text, progress_bar)
        
        # 실제 토론 완료 확인
        st.write(f"■ {ticker} 실제 토론 완료!")
        st.write(f"■ 현재가: {final.get('last_price', 'None')} {final.get('currency', 'USD')}")
        
        # 세션 상태에 실제 토론 결과 저장
        st.session_state.real_debate_data = {
            'logs': logs,
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
        
        return logs, final
        
    except Exception as e:
        st.error(f"토론 실행 중 오류 발생: {e}")
        import traceback
        st.error(f"상세 오류: {traceback.format_exc()}")
        # 오류 발생 시 빈 데이터 반환
        return [], {}


# 토론 시작 버튼
if st.sidebar.button("■ 토론 시작", type="primary"):
    # 진행 상황과 함께 토론 실행
    logs, final = run_real_debate_with_progress(ticker, rounds)
    
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
            st.metric("■ 현재가", f"{final.get('last_price', final.get('current_price', 0)):.2f}")
            st.metric("■ 예측 범위", f"{min([final.get(f'{agent}_next_close', 0) for agent in ['SentimentalAgent', 'TechnicalAgent', 'FundamentalAgent']]):.2f} ~ {max([final.get(f'{agent}_next_close', 0) for agent in ['SentimentalAgent', 'TechnicalAgent', 'FundamentalAgent']]):.2f}")
            st.metric("■ 변동성", f"{((max([final.get(f'{agent}_next_close', 0) for agent in ['SentimentalAgent', 'TechnicalAgent', 'FundamentalAgent']]) - min([final.get(f'{agent}_next_close', 0) for agent in ['SentimentalAgent', 'TechnicalAgent', 'FundamentalAgent']])) / final.get('last_price', final.get('current_price', 1)) * 100):.1f}%")
    
    # 라운드별 상세 결과 표시
    with st.expander("■ 라운드별 상세 결과", expanded=False):
        for i, log in enumerate(logs):
            st.subheader(f"■ 라운드 {log.round_no}")
            
            # 각 에이전트의 의견 표시
            for opinion in log.opinions:
                agent_name = opinion.agent_id.replace('Agent', '')
                price = opinion.target.next_close
                reason = getattr(opinion, 'reason', '') or getattr(opinion.target, 'reason', '분석 결과')
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.write(f"**{agent_name}:** {price:.2f}")
                with col2:
                    st.write(f"*{reason[:100]}{'...' if len(reason) > 100 else ''}*")
            
            # 반박/지지 결과 표시
            if log.rebuttals:
                st.write("**■ 반박/지지 결과:**")
                for rebuttal in log.rebuttals:
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
        logs = st.session_state.real_debate_data['logs']
        final = st.session_state.real_debate_data['final']
        st.success("■ 실제 토론 데이터를 사용 중입니다.")
    else:
        # 초기 상태
        logs, final = [], {}

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
    st.metric("■ 평균 예측가격", f"{final.get('mean_next_close', 0):.2f}")
with col2:
    st.metric("■ 중앙값", f"{final.get('median_next_close', 0):.2f}")
with col3:
    st.metric("■ 현재가", f"{final.get('last_price', final.get('current_price', 0)):.2f}")
with col4:
    st.metric("■ 총 라운드", len(logs))

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

# 1. 최종의견 표
with tab1:
    st.header("■ 최종의견 표")
    
    # 최종 의견 데이터 준비
    final_opinions_data = []
    for agent_id, price in final.items():
        if agent_id.endswith('_next_close'):
            agent_name = agent_id.replace('_next_close', '')
            display_name = agent_name.replace('Agent', '')
            
            # 해당 에이전트의 전체 투자의견 찾기
            total_opinion = ""
            for log in logs:
                for opinion in log.opinions:
                    if opinion.agent_id == agent_name:
                        reasoning_text = getattr(opinion, 'reasoning', None) or getattr(opinion, 'reason', '')
                        if reasoning_text and log.round_no == len(logs):
                            total_opinion = reasoning_text[:100] + "..." if len(reasoning_text) > 100 else reasoning_text
                        break
            
            final_opinions_data.append({
                '에이전트': display_name,
                '최종 예측 가격': f"{price:.2f}",
                '전체 투자의견': total_opinion
            })
    
    if final_opinions_data:
        df_final = pd.DataFrame(final_opinions_data)
        st.dataframe(df_final, use_container_width=True, hide_index=True)

# 2. 투자의견 표
with tab2:
    st.header("■ 투자의견 표")
    
    # 투자의견 데이터 준비 - 실제 에이전트 의견 표시
    if logs and len(logs) > 0:
        st.markdown("### ■ 라운드별 에이전트 의견 상세")
        
        # 각 라운드별로 상세 표시
        for i, log in enumerate(logs):
            with st.expander(f"■ Round {log.round_no} - 에이전트 의견", expanded=(i == 0)):
                st.markdown(f"**라운드 {log.round_no}에서의 각 에이전트 의견:**")
                
                # 에이전트별 의견 표시
                for opinion in log.opinions:
                    agent_name = opinion.agent_id.replace('Agent', '')
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
        agent_order = ['SentimentalAgent', 'TechnicalAgent', 'FundamentalAgent']
        
        for log in logs:
            row = {'라운드': f"Round {log.round_no}"}
            for agent_id in agent_order:
                found = False
                for opinion in log.opinions:
                    if opinion.agent_id == agent_id:
                        price = opinion.target.next_close
                        reasoning_text = getattr(opinion, 'reasoning', None) or getattr(opinion, 'reason', '')
                        if reasoning_text:
                            reasoning = reasoning_text[:80] + "..." if len(reasoning_text) > 80 else reasoning_text
                            row[agent_id.replace('Agent', '')] = f"**{price:.2f}**\n\n*{reasoning}*"
                        else:
                            row[agent_id.replace('Agent', '')] = f"**{price:.2f}**"
                        found = True
                        break
                if not found:
                    row[agent_id.replace('Agent', '')] = "-"
            opinions_data.append(row)
        
        if opinions_data:
            df_opinions = pd.DataFrame(opinions_data)
            st.dataframe(df_opinions, use_container_width=True, hide_index=True, height=400)
    else:
        st.info("투자의견 데이터가 없습니다.")

# 3. 최종 예측 비교
with tab3:
    st.header("■ 최종 예측 비교")
    
    # 최종 예측 데이터 준비
    final_agents = []
    final_prices = []
    for agent_id, price in final.items():
        if agent_id.endswith('_next_close'):
            agent_name = agent_id.replace('_next_close', '').replace('Agent', '')
            final_agents.append(agent_name)
            final_prices.append(price)
    
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

# 4. 라운드별 의견 변화
with tab4:
    st.header("■ 라운드별 의견 변화")
    
    # 라운드별 데이터 준비
    rounds_data = []
    agents_data = {}
    current_price = final.get('last_price', final.get('current_price', 0))
    
    for log in logs:
        rounds_data.append(log.round_no)
        for opinion in log.opinions:
            agent_id = opinion.agent_id
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
        if current_price > 0:
            fig.add_hline(y=current_price, line_dash="dash", line_color="black",
                         annotation_text=f"현재가: {current_price:.2f}")
        
        fig.update_layout(
            title="라운드별 에이전트 의견 변화",
            xaxis_title="라운드",
            yaxis_title="예측 가격",
            height=chart_height
        )
        
        st.plotly_chart(fig, use_container_width=True)

# 5. 반박/지지 패턴
with tab5:
    st.header("■ 반박/지지 패턴")
    
    # 반박/지지 데이터 준비
    agent_rebuttal_data = {}
    all_rebuttals = []  # 모든 반박/지지 메시지 저장
    
    for log in logs:
        for rebuttal in log.rebuttals:
            from_agent = rebuttal.from_agent_id.replace('Agent', '')
            to_agent = rebuttal.to_agent_id.replace('Agent', '')
            
            if from_agent not in agent_rebuttal_data:
                agent_rebuttal_data[from_agent] = {'REBUT': 0, 'SUPPORT': 0}
            agent_rebuttal_data[from_agent][rebuttal.stance] += 1
            
            # 상세 정보 저장
            all_rebuttals.append({
                'round': log.round_no,
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
        st.markdown("### 📈 반박/지지 요약 통계")
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

# 푸터
st.markdown("---")

# 정보 섹션
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### ■ 데이터 정보")
    st.info(f"""
    - **종목**: {ticker}
    - **라운드 수**: {rounds}
    - **에이전트 수**: 3개
    - **데이터 포인트**: {len(logs) * 3}개
    """)

with col2:
    st.markdown("### ■ 주요 지표")
    st.success(f"""
    - **예측 범위**: {min([final.get(f'{agent}_next_close', 0) for agent in ['SentimentalAgent', 'TechnicalAgent', 'FundamentalAgent']]):.2f} ~ {max([final.get(f'{agent}_next_close', 0) for agent in ['SentimentalAgent', 'TechnicalAgent', 'FundamentalAgent']]):.2f}
    - **표준편차**: {np.std([final.get(f'{agent}_next_close', 0) for agent in ['SentimentalAgent', 'TechnicalAgent', 'FundamentalAgent']]):.2f}
    - **변동성**: {'높음' if np.std([final.get(f'{agent}_next_close', 0) for agent in ['SentimentalAgent', 'TechnicalAgent', 'FundamentalAgent']]) > 5 else '낮음'}
    """)

with col3:
    st.markdown("### 💡 사용 팁")
    st.warning("""
    - 사이드바에서 종목과 라운드 수 변경
    - 차트 높이와 표시 옵션 조정
    - 데이터 새로고침 버튼으로 실시간 업데이트
    - 탭을 클릭하여 각 섹션 탐색
    """)

st.markdown("---")
st.markdown("🚀 **Streamlit 대시보드** | 📈 **실시간 데이터 시각화** | 🎨 **인터랙티브 차트**")

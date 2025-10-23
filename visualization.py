# visualization.py
import matplotlib
import os

# 백엔드 설정 - 환경에 따라 자동 선택
def setup_matplotlib_backend():
    """환경에 맞는 matplotlib 백엔드 설정"""
    # 파일 저장 전용 백엔드 사용 (headless 환경에 최적화)
    matplotlib.use('Agg')
    return False

# 백엔드 설정
GUI_AVAILABLE = setup_matplotlib_backend()
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class DebateVisualizer:
    """토론 결과 시각화 클래스"""
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        """시각화 스타일 초기화"""
        plt.style.use(style)
        sns.set_palette("husl")
        self._print_backend_status()
    
    def _print_backend_status(self):
        """현재 백엔드 상태 출력"""
        current_backend = matplotlib.get_backend()
        print(f"📊 matplotlib 백엔드: {current_backend} (파일 저장 모드)")
    
    def try_enable_gui(self):
        """GUI 백엔드 활성화 시도 (headless 환경에서는 항상 False)"""
        return False
        
    def plot_round_progression(self, logs: List, final: Dict, save_path: Optional[str] = None) -> None:
        """라운드별 의견 변화 시각화"""
        if not logs:
            print("시각화할 로그 데이터가 없습니다.")
            return
            
        # 데이터 준비
        rounds = []
        agents = {}
        agent_currencies = {}
        current_price = final.get('current_price', 0)
        currency = final.get('currency', 'USD')
        
        for log in logs:
            rounds.append(log.round_no)
            for opinion in log.opinions:
                agent_id = opinion.agent_id
                if agent_id not in agents:
                    agents[agent_id] = []
                agents[agent_id].append(opinion.target.next_close)
                
                # 통화 정보 저장 (첫 번째 라운드에서만)
                if log.round_no == 1 and agent_id not in agent_currencies:
                    agent_currencies[agent_id] = final.get('currency', 'USD')
        
        # 차트 생성
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        fig.suptitle('라운드별 투자 의견 및 종합 의견', fontsize=24, fontweight='bold', y=0.95)
        
        # 1) 라운드별 의견 변화
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for i, (agent_id, prices) in enumerate(agents.items()):
            display_name = self._get_display_name(agent_id)
            currency = agent_currencies.get(agent_id, 'USD')
            
            # hover 정보를 위한 커스텀 포맷
            hover_text = [f'{display_name}: {price:.2f} {currency}' for price in prices]
            
            line = ax1.plot(rounds, prices, marker='o', linewidth=3, markersize=8, 
                           label=display_name, color=colors[i % len(colors)])
            
            # 각 점에 hover 정보 추가 (matplotlib에서는 직접적인 hover는 제한적이지만 시각적 개선)
            for j, (x, y, text) in enumerate(zip(rounds, prices, hover_text)):
                ax1.annotate(f'{y:.2f}', (x, y), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=9, alpha=0.7)
        
        # 현재가 라인 추가
        if current_price > 0:
            ax1.axhline(y=current_price, color='black', linestyle='-', linewidth=2, alpha=0.7,
                       label=f'현재가: {current_price:.2f} {currency}')
        
        ax1.set_title('라운드별 에이전트 의견 변화', fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('라운드', fontsize=12)
        ax1.set_ylabel('예측 가격', fontsize=12)
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # y축 범위를 데이터에 맞게 조정하여 차이를 더 잘 보이게 함
        all_prices = [price for prices in agents.values() for price in prices]
        if current_price > 0:
            all_prices.append(current_price)
        price_min, price_max = min(all_prices), max(all_prices)
        price_range = price_max - price_min
        ax1.set_ylim(price_min - price_range * 0.1, price_max + price_range * 0.1)
        
        # 2) 최종 결과 비교
        final_prices = [final['agents'][agent] for agent in final['agents'].keys()]
        final_agents = [self._get_display_name(agent) for agent in final['agents'].keys()]
        
        bars = ax2.bar(final_agents, final_prices, alpha=0.8, color=colors[:len(final_agents)])
        
        # 예측 평균 라인으로 변경
        prediction_mean = final['mean_next_close']
        ax2.axhline(y=prediction_mean, color='red', linestyle='--', linewidth=2,
                   label=f'예측 평균: {prediction_mean:.2f}')
        ax2.axhline(y=final['median_next_close'], color='green', linestyle='--', linewidth=2,
                   label=f'중앙값: {final["median_next_close"]:.2f}')
        
        # 막대 위에 값 표시
        for bar, price in zip(bars, final_prices):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(final_prices) * 0.01, 
                    f'{price:.2f}', ha='center', va='bottom', fontweight='bold')
        
        ax2.set_title('최종 예측 결과 비교', fontsize=16, fontweight='bold', pad=20)
        ax2.set_ylabel('예측 가격', fontsize=12)
        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # y축 범위 조정
        final_price_min, final_price_max = min(final_prices), max(final_prices)
        final_price_range = final_price_max - final_price_min
        ax2.set_ylim(final_price_min - final_price_range * 0.2, final_price_max + final_price_range * 0.2)
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"차트가 저장되었습니다: {save_path}")
        
        # GUI가 사용 가능한 경우에만 화면에 표시
        if GUI_AVAILABLE:
            print("차트를 화면에 표시합니다...")
            plt.show()
        else:
            plt.close()  # 메모리 절약
    
    def plot_consensus_analysis(self, logs: List, final: Dict, save_path: Optional[str] = None) -> None:
        """의견 일치도 분석 시각화"""
        if not logs:
            return
            
        # 데이터 준비
        rounds = []
        consensus_scores = []
        price_ranges = []
        agent_opinions = {}
        
        for log in logs:
            rounds.append(log.round_no)
            prices = [op.target.next_close for op in log.opinions]
            agent_names = [self._get_display_name(op.agent_id) for op in log.opinions]
            
            # 에이전트별 의견 저장
            for i, (name, price) in enumerate(zip(agent_names, prices)):
                if name not in agent_opinions:
                    agent_opinions[name] = []
                agent_opinions[name].append(price)
            
            # 일치도 점수 (표준편차의 역수)
            std_dev = np.std(prices)
            consensus_score = 1 / (1 + std_dev) if std_dev > 0 else 1
            consensus_scores.append(consensus_score)
            
            # 가격 범위
            price_range = max(prices) - min(prices)
            price_ranges.append(price_range)
        
        # 차트 생성
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle('의견 일치도 및 투자 의견 분석', fontsize=24, fontweight='bold', y=0.95)
        
        # 1) 일치도 점수 (호버 정보 개선)
        line = ax1.plot(rounds, consensus_scores, marker='o', linewidth=3, markersize=8, color='#2E8B57')
        
        # 각 점에 상세한 호버 정보 추가
        for i, (round_num, score) in enumerate(zip(rounds, consensus_scores)):
            # 해당 라운드의 에이전트별 의견 정보 수집
            log = logs[i]
            agent_details = []
            for opinion in log.opinions:
                agent_name = self._get_display_name(opinion.agent_id)
                price = opinion.target.next_close
                currency = final.get('currency', 'USD')
                agent_details.append(f"{agent_name}: {price:.2f} {currency}")
            
            # 호버 텍스트 생성
            hover_text = f"라운드 {round_num}\n일치도: {score:.3f}\n" + "\n".join(agent_details)
            
            # 주석으로 호버 정보 표시 (실제로는 matplotlib의 제한으로 인해 간단한 형태)
            ax1.annotate(f'{score:.3f}', (round_num, score), textcoords="offset points", 
                        xytext=(0,15), ha='center', fontsize=9, alpha=0.8,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        ax1.set_title('라운드별 의견 일치도', fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('라운드', fontsize=12)
        ax1.set_ylabel('일치도 점수 (0-1)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # 각 점에 값 표시
        for x, y in zip(rounds, consensus_scores):
            ax1.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9, alpha=0.7)
        
        # 2) 가격 범위
        ax2.plot(rounds, price_ranges, marker='s', linewidth=3, markersize=8, color='#DC143C')
        ax2.set_title('라운드별 예측 가격 범위', fontsize=16, fontweight='bold', pad=20)
        ax2.set_xlabel('라운드', fontsize=12)
        ax2.set_ylabel('가격 범위', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 각 점에 값 표시
        for x, y in zip(rounds, price_ranges):
            ax2.annotate(f'{y:.2f}', (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9, alpha=0.7)
        
        # y축 범위 조정하여 차이를 더 잘 보이게 함
        if price_ranges:
            range_min, range_max = min(price_ranges), max(price_ranges)
            range_diff = range_max - range_min
            if range_diff > 0:
                ax2.set_ylim(range_min - range_diff * 0.1, range_max + range_diff * 0.1)
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"일치도 분석 차트가 저장되었습니다: {save_path}")
        
        # GUI가 사용 가능한 경우에만 화면에 표시
        if GUI_AVAILABLE:
            print("차트를 화면에 표시합니다...")
            plt.show()
        else:
            plt.close()  # 메모리 절약
    
    def plot_rebuttal_network(self, logs: List, save_path: Optional[str] = None) -> None:
        """반박/지지 네트워크 시각화"""
        if not logs:
            return
            
        # 반박/지지 데이터 수집
        rebuttal_data = []
        for log in logs:
            for rebuttal in log.rebuttals:
                rebuttal_data.append({
                    'from': self._get_display_name(rebuttal.from_agent_id),
                    'to': self._get_display_name(rebuttal.to_agent_id),
                    'stance': rebuttal.stance,
                    'round': log.round_no
                })
        
        if not rebuttal_data:
            print("반박/지지 데이터가 없습니다.")
            return
        
        df = pd.DataFrame(rebuttal_data)
        
        # 차트 생성 - 각 Agent별로 분리
        unique_agents = df['from'].unique()
        n_agents = len(unique_agents)
        
        if n_agents == 0:
            print("에이전트 데이터가 없습니다.")
            return
            
        # 2x2 또는 3x1 레이아웃으로 조정
        if n_agents <= 2:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
        
        fig.suptitle('에이전트별 반박/지지 패턴 분석', fontsize=24, fontweight='bold', y=0.95)
        
        # 각 에이전트별로 차트 생성
        for i, agent in enumerate(unique_agents):
            if i >= len(axes):
                break
                
            ax = axes[i]
            agent_data = df[df['from'] == agent]
            
            # 해당 에이전트의 반박/지지 패턴
            stance_counts = agent_data['stance'].value_counts()
            colors = ['#FF6B6B' if stance == 'REBUT' else '#4ECDC4' for stance in stance_counts.index]
            
            bars = ax.bar(stance_counts.index, stance_counts.values, color=colors, alpha=0.8)
            ax.set_title(f'{agent}의 반박/지지 패턴', fontsize=16, fontweight='bold', pad=20)
            ax.set_ylabel('개수', fontsize=12)
            
            # 막대 위에 값 표시
            for bar, count in zip(bars, stance_counts.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                       str(count), ha='center', va='bottom', fontweight='bold')
        
        # 빈 subplot 숨기기
        for i in range(n_agents, len(axes)):
            axes[i].set_visible(False)
        
        # 전체 반박/지지 분포 (마지막 subplot에)
        if len(axes) > n_agents:
            ax_final = axes[-1]
            round_stance = df.groupby(['round', 'stance']).size().unstack(fill_value=0)
            bars = round_stance.plot(kind='bar', ax=ax_final, stacked=True, 
                                   color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
            ax_final.set_title('라운드별 전체 반박/지지 분포', fontsize=16, fontweight='bold', pad=20)
            ax_final.set_xlabel('라운드', fontsize=12)
            ax_final.set_ylabel('개수', fontsize=12)
            ax_final.legend(title='입장', loc='center left', bbox_to_anchor=(1, 0.5), fontsize=11)
        
        # 막대 위에 값 표시 (마지막 subplot이 있는 경우에만)
        if len(axes) > n_agents:
            ax_final = axes[-1]
            for container in ax_final.containers:
                ax_final.bar_label(container, label_type='center', fontsize=9)
        
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"반박 네트워크 차트가 저장되었습니다: {save_path}")
        
        # GUI가 사용 가능한 경우에만 화면에 표시
        if GUI_AVAILABLE:
            print("차트를 화면에 표시합니다...")
            plt.show()
        else:
            plt.close()  # 메모리 절약
    
    def plot_opinion_table(self, logs: List, final: Dict, save_path: Optional[str] = None) -> None:
        """라운드별 Agent 투자의견 표와 종합의견 시각화"""
        if not logs:
            print("시각화할 로그 데이터가 없습니다.")
            return
        
        # 데이터 준비
        rounds = []
        agent_opinions = {}
        currency = final.get('currency', 'USD')
        
        # 에이전트 순서 정의
        agent_order = ['SentimentalAgent', 'FundamentalAgent', 'TechnicalAgent']
        agent_display_names = {
            'SentimentalAgent': 'Sentimental Agent',
            'FundamentalAgent': 'Fundamental Agent', 
            'TechnicalAgent': 'Technical Agent'
        }
        
        for log in logs:
            rounds.append(log.round_no)
            for opinion in log.opinions:
                agent_id = opinion.agent_id
                if agent_id not in agent_opinions:
                    agent_opinions[agent_id] = {}
                agent_opinions[agent_id][log.round_no] = opinion.target.next_close
        
        # 차트 생성
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        fig.suptitle('라운드별 투자 의견 및 종합 의견', fontsize=24, fontweight='bold', y=0.95)
        
        # 1) 라운드별 Agent 투자의견 표
        table_data = []
        for round_num in rounds:
            row = [f"Round {round_num}"]
            for agent_id in agent_order:
                if agent_id in agent_opinions and round_num in agent_opinions[agent_id]:
                    price = agent_opinions[agent_id][round_num]
                    row.append(f"**[{price:.2f} {currency}]**")
                else:
                    row.append("-")
            table_data.append(row)
        
        # 표 생성
        headers = ['라운드'] + [agent_display_names.get(agent, agent) for agent in agent_order]
        table = ax1.table(cellText=table_data, colLabels=headers, 
                         cellLoc='center', loc='center', bbox=[0.1, 0.1, 0.8, 0.8])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # 표 스타일링
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax1.set_title('라운드별 Agent 투자 의견', fontsize=16, fontweight='bold', pad=20)
        ax1.axis('off')
        
        # 2) 종합 투자의견
        final_mean = final.get('mean_next_close', 0)
        final_median = final.get('median_next_close', 0)
        
        # 종합의견 텍스트 박스
        comprehensive_text = f"**[{final_mean:.2f} {currency}]** : 종합 투자 의견 (평균)\n"
        comprehensive_text += f"**[{final_median:.2f} {currency}]** : 종합 투자 의견 (중앙값)"
        
        ax2.text(0.5, 0.5, comprehensive_text, transform=ax2.transAxes, 
                fontsize=20, ha='center', va='center', 
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8),
                weight='bold')
        ax2.set_title('종합 투자 의견', fontsize=16, fontweight='bold', pad=20)
        ax2.axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"투자의견 표가 저장되었습니다: {save_path}")
        
        # GUI가 사용 가능한 경우에만 화면에 표시
        if GUI_AVAILABLE:
            print("차트를 화면에 표시합니다...")
            plt.show()
        else:
            plt.close()  # 메모리 절약
    
    def create_interactive_dashboard(self, logs: List, final: Dict, ticker: str) -> None:
        """인터랙티브 대시보드 생성 (Plotly) - 5개 차트만 표시"""
        if not logs:
            return
            
        # 서브플롯 생성 - 5x1 레이아웃
        fig = make_subplots(
            rows=5, cols=1,
            subplot_titles=('[ 최종의견 표 ]', '[ 투자의견 표 ]', '[ 최종 예측 비교 ]', 
                          '[ 라운드별 의견 변화 ]', '[ 반박/지지 패턴 ]'),
            specs=[[{"type": "table"}],
                   [{"type": "table"}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}]]
        )
        
        # 1) 최종의견 표 (전체 투자의견 컬럼 추가)
        final_opinions = []
        for agent_id, price in final.items():
            if agent_id.endswith('_next_close'):
                agent_name = agent_id.replace('_next_close', '')
                display_name = self._get_display_name(agent_name)
                
                # 해당 에이전트의 전체 투자의견 찾기
                total_opinion = ""
                for log in logs:
                    for opinion in log.opinions:
                        if opinion.agent_id == agent_name:
                            reasoning_text = getattr(opinion, 'reasoning', None) or getattr(opinion, 'reason', '')
                            if reasoning_text:
                                # 마지막 라운드의 의견만 사용
                                if log.round_no == len(logs):
                                    total_opinion = reasoning_text[:100] + "..." if len(reasoning_text) > 100 else reasoning_text
                                    # 30자마다 줄바꿈 추가
                                    total_opinion = '<br>'.join([total_opinion[i:i+20] for i in range(0, len(total_opinion), 30)])
                            break
                
                final_opinions.append([display_name, f"{price:.2f}", total_opinion])
        
        if final_opinions:
            fig.add_trace(
                go.Table(
                    header=dict(values=['에이전트', '최종 예측 가격', '전체 투자의견'], 
                               fill_color='#2E8B57', 
                               font=dict(color='white', size=14)),  # 절반으로 줄이기 (28 -> 14)
                    cells=dict(values=list(zip(*final_opinions)), 
                              fill_color='#f0f0f0', 
                              font=dict(size=12),  # 절반으로 줄이기 (24 -> 12)
                              height=240,  # 높이는 유지
                              align=['center', 'center', 'center'])
                ),
                row=1, col=1
            )
        
        # 2) 투자의견 표 (테이블 형태) - 가격과 의견 내용 포함
        table_data = []
        table_headers = ['라운드', 'Sentimental', 'Technical', 'Fundamental']
        
        # 에이전트 순서 정의
        agent_order = ['SentimentalAgent', 'FundamentalAgent', 'TechnicalAgent']
        
        for log in logs:
            row = [f"Round {log.round_no}"]
            for agent_id in agent_order:
                found = False
                for opinion in log.opinions:
                    if opinion.agent_id == agent_id:
                        # 가격과 의견 내용을 함께 표시
                        price = opinion.target.next_close
                        # reasoning 또는 reason 속성 사용 (데이터 구조에 따라)
                        reasoning_text = getattr(opinion, 'reasoning', None) or getattr(opinion, 'reason', '')
                        if reasoning_text:
                            # 긴 텍스트를 줄바꿈으로 처리 (30자마다 줄바꿈)
                            reasoning = reasoning_text[:100] + "..." if len(reasoning_text) > 100 else reasoning_text
                            # 30자마다 줄바꿈 추가
                            reasoning_wrapped = '<br>'.join([reasoning[i:i+30] for i in range(0, len(reasoning), 30)])
                            row.append(f"{price:.2f}<br>{reasoning_wrapped}")
                        else:
                            row.append(f"{price:.2f}")
                        found = True
                        break
                if not found:
                    row.append("-")
            table_data.append(row)
        
        fig.add_trace(
            go.Table(
                header=dict(values=table_headers, fill_color='#4CAF50', 
                           font=dict(color='white', size=14)),  # 14px로 변경
                cells=dict(values=list(zip(*table_data)), 
                          fill_color='#f0f0f0', 
                          font=dict(size=12),  # 12px로 변경
                          height=360,  # 높이는 유지
                          align=['center', 'center', 'center', 'center'])  # 모두 가운데 정렬
            ),
            row=2, col=1
        )
        
        # 3) 최종 예측 비교 (예측 평균 라인 포함)
        final_agents = [self._get_display_name(agent) for agent in final['agents'].keys()]
        final_prices = [final['agents'][agent] for agent in final['agents'].keys()]
        
        fig.add_trace(
            go.Bar(x=final_agents, y=final_prices, name='최종 예측',
                  marker_color=['lightblue', 'lightcoral', 'lightgreen'],
                  showlegend=True, legendgroup="group1"),
            row=3, col=1
        )
        
        # 예측 평균을 수평 점선으로 추가
        prediction_mean = final['mean_next_close']
        
        # 평균 수평선 (Scatter로 구현)
        fig.add_trace(
            go.Scatter(x=final_agents, y=[prediction_mean] * len(final_agents), 
                      mode='lines', line=dict(dash='dot', color='red', width=2),
                      name=f'평균 : {prediction_mean:.2f}',
                      showlegend=True, legendgroup="group1"),
            row=3, col=1
        )
        
        # 각 막대 위에 금액 표기
        for i, (agent, price) in enumerate(zip(final_agents, final_prices)):
            fig.add_annotation(
                x=agent, y=price,
                text=f"{price:.2f}",
                showarrow=False,
                font=dict(size=6, color="black"),
                yshift=10,
                row=3, col=1
            )
        
        # 4) 라운드별 의견 변화 (현재가 라인 포함)
        rounds = []
        agents_data = {}
        agent_currencies = {}
        current_price = final.get('current_price', 0)
        currency = final.get('currency', 'USD')
        
        for log in logs:
            rounds.append(log.round_no)
            for opinion in log.opinions:
                agent_id = opinion.agent_id
                if agent_id not in agents_data:
                    agents_data[agent_id] = []
                agents_data[agent_id].append(opinion.target.next_close)
                
                # 통화 정보 저장
                if agent_id not in agent_currencies:
                    agent_currencies[agent_id] = final.get('currency', 'USD')
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for i, (agent_id, prices) in enumerate(agents_data.items()):
            display_name = self._get_display_name(agent_id)
            agent_currency = agent_currencies.get(agent_id, 'USD')
            
            # hover 텍스트 생성
            hover_text = [f'{display_name}: {price:.2f} {agent_currency}' for price in prices]
            
            fig.add_trace(
                go.Scatter(x=rounds, y=prices, mode='lines+markers',
                          name=display_name, line=dict(color=colors[i % len(colors)], width=3),
                          marker=dict(size=8),
                          hovertemplate='<b>%{text}</b><br>라운드: %{x}<br>예측가격: %{y:.2f}<extra></extra>',
                          text=hover_text,
                          showlegend=True, legendgroup="group2"),
                row=4, col=1
            )
            
            # 각 점 위에 금액 표기
            for j, (round_num, price) in enumerate(zip(rounds, prices)):
                fig.add_annotation(
                    x=round_num, y=price,
                    text=f"{price:.2f}",
                    showarrow=False,
                    font=dict(size=6, color=colors[i % len(colors)]),
                    yshift=15,
                    row=4, col=1
                )
        
        # 현재가를 수평선으로 추가
        if current_price > 0:
            # 최소/최대 라운드 값 찾기
            min_round = min(rounds) if rounds else 1
            max_round = max(rounds) if rounds else 3
            
            fig.add_trace(
                go.Scatter(x=[min_round, max_round], y=[current_price, current_price], 
                          mode='lines', line=dict(dash='dash', color='black', width=2),
                          name=f'최근 종가 : {current_price:.2f} {currency}',
                          showlegend=True, legendgroup="group2"),
                row=4, col=1
            )
        
        # 5) 반박/지지 패턴 (막대차트)
        agent_rebuttal_data = {}
        for log in logs:
            for rebuttal in log.rebuttals:
                from_agent = self._get_display_name(rebuttal.from_agent_id)
                if from_agent not in agent_rebuttal_data:
                    agent_rebuttal_data[from_agent] = {'REBUT': 0, 'SUPPORT': 0}
                agent_rebuttal_data[from_agent][rebuttal.stance] += 1
        
        # 모든 에이전트의 반박/지지 패턴을 막대차트로 표시
        if agent_rebuttal_data:
            agents = list(agent_rebuttal_data.keys())
            rebut_counts = [agent_rebuttal_data[agent]['REBUT'] for agent in agents]
            support_counts = [agent_rebuttal_data[agent]['SUPPORT'] for agent in agents]
            
            fig.add_trace(
                go.Bar(x=agents, y=rebut_counts, name='반박', 
                      marker_color='#FF6B6B',
                      showlegend=True, legendgroup="group3"),
                row=5, col=1
            )
            
            fig.add_trace(
                go.Bar(x=agents, y=support_counts, name='지지', 
                      marker_color='#4ECDC4',
                      showlegend=True, legendgroup="group3"),
                row=5, col=1
            )
            
            # 각 막대 위에 숫자 표기
            for i, (agent, rebut_count, support_count) in enumerate(zip(agents, rebut_counts, support_counts)):
                # 반박 막대 위에 숫자
                fig.add_annotation(
                    x=agent, y=rebut_count,
                    text=f"{rebut_count}",
                    showarrow=False,
                    font=dict(size=6, color="black"),
                    yshift=10,
                    row=5, col=1
                )
                # 지지 막대 위에 숫자
                fig.add_annotation(
                    x=agent, y=support_count,
                    text=f"{support_count}",
                    showarrow=False,
                    font=dict(size=6, color="black"),
                    yshift=10,
                    row=5, col=1
                )
                
                # 각 막대 가운데 중간에 반박/지지 표시
                fig.add_annotation(
                    x=agent, y=max(rebut_count, support_count) / 2,
                    text="반박" if rebut_count > support_count else "지지",
                    showarrow=False,
                    font=dict(size=8, color="white", family="Arial Black"),
                    row=5, col=1
                )
        
        # 레이아웃 업데이트
        fig.update_layout(
            title=dict(
                text=f'{ticker} 토론 결과 대시보드',
                x=0.5,
                xanchor='center',
                font=dict(size=24, family="Arial Black")
            ),
            showlegend=True,  # 범례 활성화
            height=2500,  # 높이 증가 (표 높이 2배 증가에 맞춰 조정)
            margin=dict(l=50, r=50, t=100, b=50)
        )
        
        # 서브플롯 타이틀 스타일 업데이트 (12px로 설정)
        fig.update_annotations(
            font=dict(size=12, family="Arial Black"),  # 12px로 설정
            font_color="black"
        )
        
        # 축 레이블 설정
        fig.update_xaxes(title_text="에이전트", row=3, col=1)
        fig.update_yaxes(title_text="예측 가격", row=3, col=1)
        fig.update_xaxes(title_text="라운드", tickmode='linear', dtick=1, row=4, col=1)
        fig.update_yaxes(title_text="예측 가격", row=4, col=1)
        fig.update_xaxes(title_text="에이전트", row=5, col=1)
        fig.update_yaxes(title_text="개수", row=5, col=1)
        
        fig.show()
    
    def plot_stock_context(self, ticker: str, period: str = "1mo", save_path: Optional[str] = None) -> None:
        """주식 컨텍스트 시각화 (yfinance 데이터)"""
        try:
            import yfinance as yf
            
            # 주식 데이터 가져오기
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            info = stock.info
            
            if hist.empty:
                print(f"{ticker} 데이터를 가져올 수 없습니다.")
                return
            
            # 차트 생성
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{ticker} 주식 컨텍스트 분석', fontsize=24, fontweight='bold', y=0.95)
            
            # 1) 주가 차트
            ax1.plot(hist.index, hist['Close'], linewidth=3, label='종가', color='#1f77b4')
            ax1.plot(hist.index, hist['Close'].rolling(20).mean(), 
                    linewidth=2, alpha=0.8, label='20일 이동평균', color='#ff7f0e')
            ax1.set_title(f'{ticker} 주가 차트 ({period})', fontsize=16, fontweight='bold', pad=20)
            ax1.set_ylabel('가격', fontsize=12)
            ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=11)
            ax1.grid(True, alpha=0.3)
            
            # 2) 거래량
            ax2.bar(hist.index, hist['Volume'], alpha=0.8, color='#ff7f0e')
            ax2.set_title('거래량', fontsize=16, fontweight='bold', pad=20)
            ax2.set_ylabel('거래량', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            # 3) 일일 수익률
            daily_returns = hist['Close'].pct_change().dropna()
            ax3.hist(daily_returns, bins=30, alpha=0.8, color='#2ca02c')
            ax3.axvline(daily_returns.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'평균: {daily_returns.mean():.3f}')
            ax3.set_title('일일 수익률 분포', fontsize=16, fontweight='bold', pad=20)
            ax3.set_xlabel('수익률', fontsize=12)
            ax3.set_ylabel('빈도', fontsize=12)
            ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=11)
            ax3.grid(True, alpha=0.3)
            
            # 4) 기본 정보
            info_text = f"""
                회사명: {info.get('longName', 'N/A')}
                섹터: {info.get('sector', 'N/A')}
                현재가: ${info.get('currentPrice', 'N/A')}
                시가총액: ${info.get('marketCap', 'N/A'):,}
                52주 최고가: ${info.get('fiftyTwoWeekHigh', 'N/A')}
                52주 최저가: ${info.get('fiftyTwoWeekLow', 'N/A')}
                배당 수익률: {info.get('dividendYield', 'N/A')}
            """
            ax4.text(0.1, 0.5, info_text, transform=ax4.transAxes, 
                    fontsize=11, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            ax4.set_title('기본 정보', fontsize=16, fontweight='bold', pad=20)
            ax4.axis('off')
            
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.3, wspace=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"주식 컨텍스트 차트가 저장되었습니다: {save_path}")
            
            # GUI가 사용 가능한 경우에만 화면에 표시
            plt.close()  # 메모리 절약
            
        except ImportError:
            print("yfinance가 설치되지 않았습니다. pip install yfinance")
        except Exception as e:
            print(f"주식 데이터 가져오기 실패: {e}")
    
    def _get_display_name(self, agent_id: str) -> str:
        """에이전트 ID를 표시명으로 변환"""
        agent_id_lower = agent_id.lower()
        if "sentiment" in agent_id_lower:
            return "Sentimental"
        elif "technical" in agent_id_lower:
            return "Technical"
        elif "fundamental" in agent_id_lower:
            return "Fundamental"
        else:
            return agent_id
    
    def generate_report(self, logs: List, final: Dict, ticker: str, 
                       save_dir: str = "./reports") -> None:
        """종합 리포트 생성"""
        import os
        
        # 리포트 디렉토리 생성
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1) 라운드별 진행
        self.plot_round_progression(
            logs, final, 
            f"{save_dir}/{ticker}_round_progression_{timestamp}.png"
        )
        
        # 2) 일치도 분석
        self.plot_consensus_analysis(
            logs, final,
            f"{save_dir}/{ticker}_consensus_{timestamp}.png"
        )
        
        # 3) 반박 네트워크
        self.plot_rebuttal_network(
            logs,
            f"{save_dir}/{ticker}_rebuttal_network_{timestamp}.png"
        )
        
        # 4) 투자의견 표
        self.plot_opinion_table(
            logs, final,
            f"{save_dir}/{ticker}_opinion_table_{timestamp}.png"
        )
        
        # 5) 주식 컨텍스트
        self.plot_stock_context(
            ticker, period="1mo",
            save_path=f"{save_dir}/{ticker}_context_{timestamp}.png"
        )
        
        # 6) 인터랙티브 대시보드
        self.create_interactive_dashboard(logs, final, ticker)
        
        print(f"모든 리포트가 {save_dir}에 저장되었습니다.")


# 사용 예제
if __name__ == "__main__":
    # 예제 데이터로 테스트
    from dataclasses import dataclass
    from typing import List
    
    @dataclass
    class MockOpinion:
        agent_id: str
        target: object
        reason: str
    
    @dataclass
    class MockTarget:
        next_close: float
    
    @dataclass
    class MockLog:
        round_no: int
        opinions: List[MockOpinion]
        rebuttals: List
    
    # 테스트 데이터 생성
    mock_logs = [
        MockLog(1, [
            MockOpinion("SentimentalAgent", MockTarget(100.0), "긍정적"),
            MockOpinion("TechnicalAgent", MockTarget(105.0), "상승 추세"),
            MockOpinion("FundamentalAgent", MockTarget(98.0), "가치 평가")
        ], []),
        MockLog(2, [
            MockOpinion("SentimentalAgent", MockTarget(102.0), "수정된 의견"),
            MockOpinion("TechnicalAgent", MockTarget(103.0), "수정된 의견"),
            MockOpinion("FundamentalAgent", MockTarget(100.0), "수정된 의견")
        ], [])
    ]
    
    mock_final = {
        "ticker": "AAPL",
        "agents": {
            "SentimentalAgent": 102.0,
            "TechnicalAgent": 103.0,
            "FundamentalAgent": 100.0
        },
        "mean_next_close": 101.67,
        "median_next_close": 102.0,
        "currency": "USD",
        "last_price": 99.5
    }
    
    # 시각화 테스트
    visualizer = DebateVisualizer()
    visualizer.plot_round_progression(mock_logs, mock_final)
    visualizer.create_interactive_dashboard(mock_logs, mock_final, "AAPL")

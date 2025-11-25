
import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import torch

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.technical_agent import TechnicalAgent
from agents.macro_agent import MacroAgent
from agents.sentimental_agent import SentimentalAgent
from config.agents import dir_info

# 1. TechnicalAgent 데이터 로더 helper
from core.technical_classes.technical_data_set import load_dataset as load_dataset_tech

def generate_ensemble_data(ticker="NVDA", days=365, output_path="data/processed/ensemble_train.csv"):
    """
    각 에이전트의 과거 예측값과 신뢰도를 수집하여 앙상블 학습 데이터 생성
    """
    print(f"[{datetime.now()}] 데이터 생성 시작: {ticker}, days={days}")
    
    # ----------------------------------------------------------------
    # 1. 에이전트 초기화
    # ----------------------------------------------------------------
    print("1. 에이전트 초기화 중...")
    
    # TechnicalAgent
    tech_agent = TechnicalAgent(ticker=ticker)
    if os.path.exists(os.path.join(dir_info["model_dir"], f"{ticker}_TechnicalAgent.pt")):
        try:
            tech_agent.load_model(os.path.join(dir_info["model_dir"], f"{ticker}_TechnicalAgent.pt"))
        except:
             pass 
    else:
        print("TechnicalAgent 모델이 없습니다. Pretrain 실행...")
        tech_agent.pretrain()
        
    # MacroAgent
    macro_agent = MacroAgent(ticker=ticker, base_date=datetime.today())
    if not macro_agent.model_path or not os.path.exists(macro_agent.model_path):
         print("MacroAgent 모델이 없습니다. Pretrain 실행...")
         macro_agent.pretrain()
    else:
         # macro_agent.load_assets() # Removed method
         macro_agent.load_model() # Use load_model instead
         
    # SentimentalAgent
    senti_agent = SentimentalAgent(ticker=ticker)
    if os.path.exists(os.path.join(dir_info["model_dir"], f"{ticker}_SentimentalAgent.pt")):
        # SentimentalAgent.predict 내부에서 로드함
        pass
    else:
         print("SentimentalAgent 모델이 없습니다. Pretrain 실행...")
         senti_agent.pretrain()
         
    print("에이전트 초기화 완료.")

    # ----------------------------------------------------------------
    # 2. 전체 데이터 로드 (효율성)
    # ----------------------------------------------------------------
    
    # 2-1. TechnicalAgent Data
    tech_dataset_path = os.path.join(dir_info["data_dir"], f"{ticker}_TechnicalAgent_dataset.csv")
    if not os.path.exists(tech_dataset_path):
        tech_agent.searcher(ticker) 
        
    tech_X_all, tech_y_all, tech_cols, tech_dates = load_dataset_tech(
        ticker, agent_id="TechnicalAgent", save_dir=dir_info["data_dir"]
    )
    
    # tech_dates 구조 확인 및 평탄화
    tech_last_dates = []
    if tech_dates is not None and len(tech_dates) > 0:
        # Case 1: tech_dates = [['2023-01-01', ...], ...]
        if isinstance(tech_dates[0], (list, tuple, np.ndarray)):
             tech_last_dates = [d[-1] for d in tech_dates]
        # Case 2: tech_dates = ['2023-01-01', ...]
        elif isinstance(tech_dates[0], str):
             tech_last_dates = tech_dates # 윈도우가 아니라 마지막 날짜 리스트라고 가정
             
    # str -> datetime 변환
    tech_last_dates_dt = pd.to_datetime(tech_last_dates).normalize()

    # 2-2. MacroAgent Data
    macro_agent.searcher(ticker) 
    macro_full_df = macro_agent.macro_df 
    macro_full_df['Date'] = pd.to_datetime(macro_full_df['Date']).dt.normalize()
    
    # 2-3. SentimentalAgent Data
    senti_sd = senti_agent.run_dataset(days=days+365)
    senti_raw = senti_sd.raw_df
    senti_raw['date'] = pd.to_datetime(senti_raw['date']).dt.normalize()
    
    # Sentimental feature columns
    senti_cols = senti_sd.feature_cols
    senti_feat_df = senti_raw[senti_cols].astype(float)
    senti_feat_values = senti_feat_df.values # (N_total, F)

    # ----------------------------------------------------------------
    # 3. 타겟 날짜 설정 (실제 가격 데이터 기준)
    # ----------------------------------------------------------------
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days + 60)
    
    print(f"2. 가격 데이터 다운로드 ({start_date.date()} ~ {end_date.date()})...")
    df_price = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if isinstance(df_price.columns, pd.MultiIndex):
        df_price.columns = [c[0] for c in df_price.columns]
    
    df_price = df_price.reset_index()
    df_price['Date'] = pd.to_datetime(df_price['Date']).dt.normalize()
    df_price = df_price.sort_values('Date')
    
    target_dates = df_price['Date'].iloc[-days:].tolist()
    
    results = []
    
    print(f"3. 일별 예측 수행 ({len(target_dates)}일)...")
    
    w_tech = tech_agent.window_size
    w_macro = macro_agent.window_size
    w_senti = senti_agent.window_size
    
    total_steps = len(target_dates)
    
    for i, curr_date in enumerate(target_dates):
        if i % 10 == 0:
            print(f"  Processing {i}/{total_steps}: {curr_date.date()}")
            
        # df_price에서 curr_date의 인덱스 및 가격
        price_row = df_price[df_price['Date'] == curr_date]
        if price_row.empty:
            continue
        idx_price = price_row.index[0]
        curr_close = float(price_row['Close'].iloc[0])
            
        # T+1 시점 (Next Close) 존재 여부 확인
        if idx_price + 1 >= len(df_price):
            continue
            
        next_close_actual = float(df_price.iloc[idx_price + 1]['Close'])
        
        # -------------------------------------
        # 3-1. Technical Prediction
        # -------------------------------------
        try:
            # tech_last_dates_dt에서 curr_date와 일치하는 인덱스 찾기
            matches = np.where(tech_last_dates_dt == curr_date)[0]
            if len(matches) > 0:
                t_idx = matches[0]
                X_batch = tech_X_all[t_idx] # (Win, F)
                X_in = np.expand_dims(X_batch, axis=0) # (1, Win, F)
                
                target_tech = tech_agent.predict(X_in, current_price=curr_close)
                pred_tech = target_tech.next_close
                conf_tech = target_tech.confidence
                unc_tech = target_tech.uncertainty
            else:
                # 날짜 불일치 -> None 처리
                pred_tech = np.nan; conf_tech = 0; unc_tech = 0
        except Exception as e:
            pred_tech = np.nan; conf_tech = 0; unc_tech = 0

        # -------------------------------------
        # 3-2. Macro Prediction
        # -------------------------------------
        try:
            m_match = macro_full_df[macro_full_df['Date'] == curr_date]
            if not m_match.empty:
                m_idx = m_match.index[0]
                if m_idx >= w_macro - 1: 
                    feat_cols = list(macro_agent.scaler_X.feature_names_in_)
                    df_slice = macro_full_df.iloc[m_idx - w_macro + 1 : m_idx + 1]
                    
                    X_slice = pd.DataFrame(index=df_slice.index)
                    for c in feat_cols:
                        if c in df_slice.columns:
                            X_slice[c] = df_slice[c]
                        else:
                            X_slice[c] = 0.0
                    
                    X_sc = macro_agent.scaler_X.transform(X_slice)
                    X_in = np.expand_dims(X_sc, axis=0)
                    X_tensor = torch.FloatTensor(X_in).to(macro_agent.device)
                    
                    target_macro = macro_agent.predict(X_tensor, current_price=curr_close)
                    pred_macro = target_macro.next_close
                    conf_macro = target_macro.confidence
                    unc_macro = target_macro.uncertainty
                else:
                     pred_macro = np.nan; conf_macro = 0; unc_macro = 0
            else:
                 pred_macro = np.nan; conf_macro = 0; unc_macro = 0
        except Exception as e:
            pred_macro = np.nan; conf_macro = 0; unc_macro = 0

        # -------------------------------------
        # 3-3. Sentimental Prediction
        # -------------------------------------
        try:
            s_match = senti_raw[senti_raw['date'] == curr_date]
            if not s_match.empty:
                s_idx = s_match.index[0]
                if s_idx >= w_senti - 1:
                    X_batch = senti_feat_values[s_idx - w_senti + 1 : s_idx + 1]
                    X_in = np.expand_dims(X_batch, axis=0)
                    
                    target_senti = senti_agent.predict(X_in, current_price=curr_close)
                    pred_senti = target_senti.next_close
                    conf_senti = target_senti.confidence
                    unc_senti = target_senti.uncertainty
                else:
                    pred_senti = np.nan; conf_senti = 0; unc_senti = 0
            else:
                pred_senti = np.nan; conf_senti = 0; unc_senti = 0
        except Exception as e:
            pred_senti = np.nan; conf_senti = 0; unc_senti = 0

        # 결과 저장
        row = {
            "Date": curr_date,
            "Last_Close": curr_close,
            "Next_Close": next_close_actual,
            
            "Tech_Pred": pred_tech,
            "Tech_Conf": conf_tech,
            "Tech_Unc": unc_tech,
            
            "Macro_Pred": pred_macro,
            "Macro_Conf": conf_macro,
            "Macro_Unc": unc_macro,
            
            "Senti_Pred": pred_senti,
            "Senti_Conf": conf_senti,
            "Senti_Unc": unc_senti
        }
        results.append(row)
        
    # 4. CSV 저장
    df_out = pd.DataFrame(results)
    print(f"4. 데이터 생성 완료: {len(df_out)}행")
    
    # 결측치 제거
    df_final = df_out.dropna()
    print(f"   결측치 제거 후: {len(df_final)}행")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)
    print(f"   저장 완료: {output_path}")

if __name__ == "__main__":
    generate_ensemble_data()

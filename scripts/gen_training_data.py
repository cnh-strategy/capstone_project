
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
from config.agents import dir_info, common_params, agents_info

# 1. TechnicalAgent 데이터 로더 helper
from core.technical_classes.technical_data_set import load_dataset as load_dataset_tech

def generate_ensemble_data(ticker="NVDA", days=None, output_path="data/processed/ensemble_train.csv"):
    """
    각 에이전트의 과거 예측값과 신뢰도를 수집하여 앙상블 학습 데이터 생성
    
    Args:
        ticker: 종목 코드
        days: 학습 기간 (일수). None이면 config/agents의 period 사용
        output_path: 출력 CSV 경로
    """
    # days가 None이면 config의 period 사용
    if days is None:
        period_str = common_params.get("period", "2y")
        # period 문자열을 일수로 변환
        if period_str.endswith("y"):
            years = int(period_str[:-1])
            days = years * 365
        elif period_str.endswith("m"):
            months = int(period_str[:-1])
            days = months * 30
        elif period_str.endswith("d"):
            days = int(period_str[:-1])
        else:
            days = 2 * 365  # 기본값: 2년
    
    print(f"[{datetime.now()}] 데이터 생성 시작: {ticker}, days={days}")
    
    # ----------------------------------------------------------------
    # 1. 에이전트 초기화
    # ----------------------------------------------------------------
    print("1. 에이전트 초기화 중...")
    
    # TechnicalAgent
    tech_cfg = agents_info.get("TechnicalAgent", {})
    tech_agent = TechnicalAgent(
        agent_id="TechnicalAgent",
        ticker=ticker,
        gamma=tech_cfg.get("gamma", 0.3),
        delta_limit=tech_cfg.get("delta_limit", 0.05)
    )
    tech_model_path = os.path.join(dir_info["model_dir"], f"{ticker}_TechnicalAgent.pt")
    if os.path.exists(tech_model_path):
        try:
            tech_agent.load_model(tech_model_path)
        except:
            pass
    else:
        print("TechnicalAgent 모델이 없습니다. Pretrain 실행...")
        tech_agent.pretrain()
        
    # MacroAgent
    macro_cfg = agents_info.get("MacroAgent", {})
    macro_window = macro_cfg.get("window_size", 40)
    macro_agent = MacroAgent(
        agent_id="MacroAgent",
        ticker=ticker,
        base_date=datetime.today(),
        window=macro_window,
        gamma=macro_cfg.get("gamma", 0.5),
        delta_limit=macro_cfg.get("delta_limit", 0.1)
    )
    macro_model_path = os.path.join(dir_info["model_dir"], f"{ticker}_MacroAgent.pt")
    macro_scaler_x_path = os.path.join(dir_info["model_dir"], "scalers", f"{ticker}_MacroAgent_xscaler.pkl")
    
    if os.path.exists(macro_model_path) and os.path.exists(macro_scaler_x_path):
        try:
            macro_agent.load_model()
            # MacroAgent는 load_model에서 스케일러를 로드하지 않으므로 수동 로드 필요할 수 있음
            # 하지만 predict() 내부에서 로드 로직이 있으므로, 파일 존재 여부만 확인하면 됨
            print("MacroAgent 모델 및 스케일러 확인 완료")
        except:
            pass
    else:
        print("MacroAgent 모델 또는 스케일러가 없습니다. Pretrain 실행...")
        macro_agent.pretrain()
         
    # SentimentalAgent
    sent_cfg = agents_info.get("SentimentalAgent", {})
    senti_agent = SentimentalAgent(
        ticker=ticker,
        agent_id="SentimentalAgent",
        gamma=sent_cfg.get("gamma", 0.3),
        delta_limit=sent_cfg.get("delta_limit", 0.05)
    )
    senti_model_path = os.path.join(dir_info["model_dir"], f"{ticker}_SentimentalAgent.pt")
    if os.path.exists(senti_model_path):
        try:
            senti_agent.load_model(senti_model_path)
        except:
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
    # macro_df는 None이므로 raw CSV를 직접 읽어서 사용
    macro_raw_path = os.path.join(os.path.dirname(dir_info["data_dir"]), "raw", f"{ticker}_MacroAgent_raw.csv")
    if not os.path.exists(macro_raw_path):
        raise FileNotFoundError(f"MacroAgent raw CSV not found: {macro_raw_path}")
    macro_full_df = pd.read_csv(macro_raw_path)
    # Date 컬럼이 문자열이면 datetime으로 변환
    if 'Date' in macro_full_df.columns:
        if macro_full_df['Date'].dtype == 'object':
            macro_full_df['Date'] = pd.to_datetime(macro_full_df['Date'], errors='coerce')
        macro_full_df['Date'] = pd.to_datetime(macro_full_df['Date']).dt.normalize()
    
    # 2-3. SentimentalAgent Data
    # window_size(40일)를 위한 여유분만 추가
    senti_window = agents_info.get("SentimentalAgent", {}).get("window_size", 40)
    senti_sd = senti_agent.run_dataset(days=days + senti_window + 30)  # window + 여유분 30일
    
    # run_dataset 반환값 None 체크
    if senti_sd is None:
        raise ValueError("SentimentalAgent.run_dataset() returned None")
    if not hasattr(senti_sd, 'raw_df') or senti_sd.raw_df is None:
        raise ValueError("SentimentalAgent StockData.raw_df is None")
    if not hasattr(senti_sd, 'feature_cols') or senti_sd.feature_cols is None:
        raise ValueError("SentimentalAgent StockData.feature_cols is None")
    
    senti_raw = senti_sd.raw_df
    # date 컬럼이 문자열이면 datetime으로 변환
    if 'date' in senti_raw.columns:
        if senti_raw['date'].dtype == 'object':
            senti_raw['date'] = pd.to_datetime(senti_raw['date'], errors='coerce')
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
            # tech_last_dates_dt에서 curr_date와 일치하는 인덱스 찾기 (날짜 형식 통일)
            curr_date_normalized = pd.to_datetime(curr_date).normalize()
            matches = np.where(tech_last_dates_dt == curr_date_normalized)[0]
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
            if i < 5:  # 처음 몇 개만 오류 로그
                print(f"    [ERROR] TechnicalAgent 예측 실패: {e}")

        # -------------------------------------
        # 3-2. Macro Prediction (다른 에이전트와 동일하게 원본 데이터 전달)
        # -------------------------------------
        try:
            # 날짜 형식 통일
            curr_date_normalized = pd.to_datetime(curr_date).normalize()
            m_match = macro_full_df[macro_full_df['Date'] == curr_date_normalized]
            if not m_match.empty:
                m_idx = m_match.index[0]
                if m_idx >= w_macro - 1: 
                    # 원본 데이터 추출 (predict 내부에서 스케일링 처리)
                    df_slice = macro_full_df.iloc[m_idx - w_macro + 1 : m_idx + 1]
                    
                    # 숫자형 컬럼만 추출 (sample_id, time_step, target, date 제외)
                    feat_cols = [c for c in df_slice.columns if c not in ['sample_id', 'time_step', 'target', 'date', 'Date']]
                    feat_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df_slice[c])]
                    
                    # 윈도우 데이터 추출 (T, F) 형태 - predict 내부에서 스케일링
                    X_values = df_slice[feat_cols].values
                    
                    # predict에 원본 데이터 전달 (내부에서 스케일링 처리)
                    target_macro = macro_agent.predict(X_values, current_price=curr_close)
                    pred_macro = target_macro.next_close
                    conf_macro = target_macro.confidence
                    unc_macro = target_macro.uncertainty
                else:
                     pred_macro = np.nan; conf_macro = 0; unc_macro = 0
            else:
                 pred_macro = np.nan; conf_macro = 0; unc_macro = 0
        except Exception as e:
            pred_macro = np.nan; conf_macro = 0; unc_macro = 0
            if i < 5:  # 처음 몇 개만 오류 로그
                print(f"    [ERROR] MacroAgent 예측 실패: {e}")

        # -------------------------------------
        # 3-3. Sentimental Prediction
        # -------------------------------------
        try:
            # 날짜 형식 통일
            curr_date_normalized = pd.to_datetime(curr_date).normalize()
            s_match = senti_raw[senti_raw['date'] == curr_date_normalized]
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
            if i < 5:  # 처음 몇 개만 오류 로그
                print(f"    [ERROR] SentimentalAgent 예측 실패: {e}")

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
        
        # 디버깅: 처음 몇 개만 상세 로그
        if i < 5:
            tech_str = f"{pred_tech:.2f}" if not np.isnan(pred_tech) else "NaN"
            macro_str = f"{pred_macro:.2f}" if not np.isnan(pred_macro) else "NaN"
            senti_str = f"{pred_senti:.2f}" if not np.isnan(pred_senti) else "NaN"
            print(f"  [DEBUG {i}] Date={curr_date.date()}, Tech={tech_str}, Macro={macro_str}, Senti={senti_str}")
        
    # 4. CSV 저장
    df_out = pd.DataFrame(results)
    print(f"4. 데이터 생성 완료: {len(df_out)}행")
    
    # 디버깅: 예측 성공률 확인
    if len(df_out) > 0:
        tech_success = df_out['Tech_Pred'].notna().sum()
        macro_success = df_out['Macro_Pred'].notna().sum()
        senti_success = df_out['Senti_Pred'].notna().sum()
        print(f"   예측 성공률: Tech={tech_success}/{len(df_out)} ({tech_success/len(df_out)*100:.1f}%), "
              f"Macro={macro_success}/{len(df_out)} ({macro_success/len(df_out)*100:.1f}%), "
              f"Senti={senti_success}/{len(df_out)} ({senti_success/len(df_out)*100:.1f}%)")
    
    # 결측치 제거: 필수 컬럼만 체크
    # 최소 2개 이상의 에이전트 예측이 있어야 유효한 데이터로 간주
    required_cols = ['Last_Close', 'Next_Close']  # 필수: 가격 정보
    pred_cols = ['Tech_Pred', 'Macro_Pred', 'Senti_Pred']
    
    # 최소 2개 이상의 예측이 있는 행만 유지
    df_out['valid_pred_count'] = df_out[pred_cols].notna().sum(axis=1)
    df_final = df_out[df_out['valid_pred_count'] >= 2].drop(columns=['valid_pred_count'])
    
    print(f"   결측치 제거 후: {len(df_final)}행 (최소 2개 이상의 에이전트 예측 필요)")
    
    if len(df_final) == 0:
        print(f"   [WARN] 유효한 데이터가 없습니다. 예측 실패 원인을 확인하세요.")
        # 원본 데이터 일부 저장 (디버깅용)
        debug_path = output_path.replace('.csv', '_debug.csv')
        df_out.to_csv(debug_path, index=False)
        print(f"   디버깅용 원본 데이터 저장: {debug_path}")
    
    # 저장 디렉토리 생성
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1) 기존 output_path로 저장 (ensemble_train.csv)
    df_final.to_csv(output_path, index=False)
    print(f"   저장 완료: {output_path}")
    
    # 2) ensemble_dataset.csv로도 저장 (입력 데이터셋)
    dataset_path = os.path.join(output_dir, f"{ticker}_ensemble_dataset.csv")
    df_final.to_csv(dataset_path, index=False)
    print(f"   저장 완료: {dataset_path}")

if __name__ == "__main__":
    generate_ensemble_data()

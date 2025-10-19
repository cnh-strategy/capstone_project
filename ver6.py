# ----------------------------------------------------
# ⭐ Colab 환경 설정: Google Drive 마운트
# ----------------------------------------------------
from google.colab import drive
import os
import sys

# ⭐⭐ 필수 수정: Google Drive 경로 설정
# 데이터를 저장한 실제 Drive 폴더 경로로 바꿔주세요.
base_path = '/content/drive/MyDrive/capstone'

try:
    drive.mount('/content/drive')
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        print(f"Drive 경로 {base_path}가 생성되었습니다.")
except Exception as e:
    print(f"Google Drive 마운트 오류 또는 폴더 생성 오류: {e}")

# 라이브러리
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------------
# --- 데이터 불러오기 및 전처리 (학습 데이터) ---
# ----------------------------------------------------

# Google Drive 경로에서 파일 로드
stock_file = os.path.join(base_path, 'stock_data.csv')
news_file = os.path.join(base_path, 'news_data2.csv')

try:
    stock_df = pd.read_csv(stock_file)
    news_df = pd.read_csv(news_file)
    print(f"✅ 데이터 로드 성공: {stock_file}, {news_file}")
except FileNotFoundError:
    print(f"오류: Drive 경로({base_path})에서 데이터 파일을 찾을 수 없습니다. 경로와 파일명을 확인해주세요.")
    sys.exit()

stock = stock_df.copy()
news = news_df.copy()

# 뉴스 데이터 전처리 및 EODhd 감성 점수 준비
news['summary'] = news['summary'].fillna('')
news = news.dropna(subset=['date'])
news['date'] = pd.to_datetime(news['date'])
news['sentiment_score'] = pd.to_numeric(news['sentiment_score'], errors='coerce').fillna(0.0)


# FinBERT 모델 및 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')
finbert = AutoModelForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
news['text'] = news['title'] + ' ' + news['summary']

# FinBERT를 사용하여 감성 점수를 계산하는 함수 (GPU 활용)
def finbert_sentiment_scores(texts, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    finbert.to(device)

    finbert.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            inputs = tokenizer(list(batch_texts), return_tensors='pt',
                               padding=True, truncation=True, max_length=512)

            inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

            outputs = finbert(**inputs)
            scores.extend(torch.softmax(outputs.logits, dim=1).cpu().numpy())

    return np.array(scores)

# 뉴스 데이터에 FinBERT 감성 점수(긍정/부정/중립 확률) 추가
print("--- FinBERT를 사용한 감성 점수 계산 시작 (GPU 활용) ---")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"현재 FinBERT 감성 분석 장치: {device}")
news[['prob_positive', 'prob_negative', 'prob_neutral']] = finbert_sentiment_scores(news['text'].values)
print("--- 감성 점수 계산 완료 ---")
news['date'] = pd.to_datetime(news['date']).dt.normalize()

# 일별/종목별 평균 감성 점수 및 뉴스 기사 수 계산
daily_sentiments = (news.groupby(['date','ticker'])
    .agg(prob_positive=('prob_positive','mean'),
         prob_negative=('prob_negative','mean'),
         prob_neutral=('prob_neutral','mean'),
         n_news=('title','count'),
         eod_sentiment=('sentiment_score', 'mean')).reset_index())

# 기술적 분석 지표(Feature)를 추가하는 함수
def add_tech_features(df):
    df['ma5']     = df['Close'].rolling(window=5).mean()
    df['ma10']    = df['Close'].rolling(window=10).mean()
    df['ret']     = df['Close'].pct_change()
    df['ret_next']      = df['ret'].shift(-1)
    return df

# 주가 데이터에 기술적 지표 추가 및 전처리
stock['Date'] = pd.to_datetime(stock['Date']).dt.normalize()
stock = stock.sort_values(['Symbol','Date']).groupby('Symbol', group_keys=False).apply(add_tech_features)
stock['next_close'] = stock.groupby('Symbol')['Close'].shift(-1)
stock = stock.dropna(subset=['ret','next_close'])


# ----------------------------------------------------------------------
# ⭐⭐ Colab UTC 시간대 오류 해결을 위한 코드 추가 (문제 해결 핵심) ⭐⭐
# ----------------------------------------------------------------------
# 병합 전 날짜 컬럼의 시간대 정보를 제거하여 통일
if stock['Date'].dt.tz is not None:
    stock['Date'] = stock['Date'].dt.tz_localize(None)

if daily_sentiments['date'].dt.tz is not None:
    daily_sentiments['date'] = daily_sentiments['date'].dt.tz_localize(None)
print("✅ 날짜 컬럼의 UTC 시간대 정보 통일 완료.")
# ----------------------------------------------------------------------


# 주가 데이터와 일별 감성 데이터를 병합
data = pd.merge(stock, daily_sentiments, how='left',
    left_on=['Date','Symbol'], right_on=['date','ticker'])

for c in ['prob_positive','prob_negative','prob_neutral','n_news', 'eod_sentiment']:
    if c in data: data[c] = data[c].fillna(0.0)

# Google Drive 경로에 파일 저장
processed_file = os.path.join(base_path, 'processed_data_with_both_sentiment_V2.csv') # ⭐ 파일명 V2로 변경
try:
    data.to_csv(processed_file, index=False)
    print(f"✅ 감성 분석이 완료된 데이터 ({processed_file}) 저장 완료")
except Exception as e:
    print(f"데이터 저장 중 오류 발생: {e}")

# LSTM 모델의 입력으로 사용할 특성(Feature) 리스트 정의
FEATURES = ['prob_positive','prob_negative','prob_neutral','n_news','ret','Close', 'eod_sentiment']

# 시계열 데이터(시퀀스) 생성 및 데이터셋 준비 함수 (기존과 동일)
def create_sequences(df, window_size=10):
    df = df.sort_values('Date')
    arr = df[FEATURES].values
    tgt = df['next_close'].values
    seqs, tgts = [], []
    for i in range(len(df) - window_size):
        seqs.append(arr[i:i+window_size])
        tgts.append(tgt[i+window_size])
    return np.array(seqs), np.array(tgts)

def prepare_dataset(df, window_size=10):
    X, y, symbol_all, date_all = [], [], [], []
    for symbol in df['Symbol'].unique():
        sd = df[df['Symbol']==symbol].sort_values('Date')
        seq, tgt = create_sequences(sd, window_size)
        if len(seq) > 0:
            X.append(seq)
            y.append(tgt)
            symbol_all.extend([symbol] * len(seq))
            date_all.extend(sd['Date'].iloc[window_size:].values)
    return (np.concatenate(X), np.concatenate(y), symbol_all, date_all) if X else (np.array([]), np.array([]), [], [])

window_size = 10

# ⭐⭐ 수정: 학습 데이터 기간을 2020-01-01부터 2024-10-31까지로 확장
train_df = data[(data['Date']>=pd.to_datetime('2020-01-01')) & (data['Date']<=pd.to_datetime('2024-10-31'))]
val_df = data[(data['Date']>=pd.to_datetime('2024-11-01')) & (data['Date']<=pd.to_datetime('2024-12-31'))]
pred_df = data.copy()

# 학습 및 검증 데이터셋 준비
X_train, y_train, symbol_train, date_train = prepare_dataset(train_df, window_size)
X_val, y_val, symbol_val, date_val = prepare_dataset(val_df, window_size)
X_pred_all, y_pred_all, symbol_pred_all, date_pred_all = prepare_dataset(pred_df, window_size)

print(f"\n>> 데이터셋 정보:")
print(f"   확장된 학습 데이터 시퀀스 수: {len(X_train)}")
print(f"   검증 데이터 시퀀스 수: {len(X_val)}")


# 데이터 표준화 함수 (기존과 동일)
def standardize(X, y, scaler_x=None, scaler_y=None, fit_mode=True):
    nsamples, nwin, nfeat = X.shape

    if fit_mode:
        scaler_x = StandardScaler().fit(X.reshape(-1, nfeat))
    X_scaled = scaler_x.transform(X.reshape(-1, nfeat)).reshape(nsamples, nwin, nfeat)

    if fit_mode:
        scaler_y = StandardScaler().fit(y.reshape(-1, 1))
    y_scaled = scaler_y.transform(y.reshape(-1, 1)).flatten()

    return X_scaled, y_scaled, scaler_x, scaler_y

X_train_scaled, y_train_scaled, scaler_x, scaler_y = standardize(X_train, y_train, fit_mode=True)
X_val_scaled, y_val_scaled, _, _ = standardize(X_val, y_val, scaler_x, scaler_y, fit_mode=False)
X_pred_all_scaled, y_pred_all_scaled, _, _ = standardize(X_pred_all, y_pred_all, scaler_x, scaler_y, fit_mode=False)

# PyTorch Dataset 및 DataLoader 설정 (기존과 동일)
class StockDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X, self.y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_loader = torch.utils.data.DataLoader(StockDataset(X_train_scaled, y_train_scaled), batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(StockDataset(X_val_scaled, y_val_scaled), batch_size=32, shuffle=False)
pred_all_loader = torch.utils.data.DataLoader(StockDataset(X_pred_all_scaled, y_pred_all_scaled), batch_size=32, shuffle=False)

# --- LSTM 모델 정의 ---
class StockSentimentLSTM(nn.Module):
    # ⭐ 수정: hidden_dim=128, num_layers=2로 구조 강화
    def __init__(self, hidden_dim=128, num_layers=2, input_size=len(FEATURES)):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze()

# 모델 학습 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n>> LSTM 학습 장치: {device} (GPU 사용)")
model = StockSentimentLSTM().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)

# --- 모델 학습 함수 --- (기존과 동일)
def train_model(train_loader, val_loader, model, criterion, optimizer, device, epochs=20):
    # (학습 로직 생략)
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        t1, train_loss = time.time(), 0
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                val_loss += criterion(model(inputs), targets).item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - Time: {time.time()-t1:.1f}s")
    return train_losses, val_losses

# 모델 학습 실행
train_losses, val_losses = train_model(train_loader, val_loader, model, criterion, optimizer, device)

# --- 모델 및 스케일러 저장 (Drive 경로) ---
torch.save(model.state_dict(), os.path.join(base_path, "model_lstm_bothsentiment_V2.pt"))
joblib.dump(scaler_x, os.path.join(base_path, "scaler_x_both_V2.pkl"))
joblib.dump(scaler_y, os.path.join(base_path, "scaler_y_both_V2.pkl"))

print("모델과 스케일러 저장 완료")

# --- 모델/스케일러 불러오기 및 예측 (Drive 경로) ---
model_eval = StockSentimentLSTM(hidden_dim=128, num_layers=2, input_size=len(FEATURES)) # ⭐ 수정된 구조 반영
model_eval.load_state_dict(torch.load(os.path.join(base_path, "model_lstm_bothsentiment_V2.pt")))
model_eval.eval()
model_eval.to(device)
scaler_x = joblib.load(os.path.join(base_path, "scaler_x_both_V2.pkl"))
scaler_y = joblib.load(os.path.join(base_path, "scaler_y_both_V2.pkl"))

# 예측 결과를 시각화하는 함수
def plot_predictions(model, val_loader, scaler_y):
    # (함수 내용 생략)
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for inputs, y_true in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds.extend(outputs.cpu().numpy())
            targets.extend(y_true.numpy())

    preds = scaler_y.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
    targets = scaler_y.inverse_transform(np.array(targets).reshape(-1,1)).flatten()

    plt.figure(figsize=(12, 6))
    plt.plot(targets, label='Actual Next Close', linewidth=2)
    plt.plot(preds, label='Predicted Next Close', linewidth=2, alpha=0.8)
    plt.title('Actual vs Predicted Next Day Close (Validation set - V2 Reinforced)')
    plt.xlabel('Sample Index (Time-ordered)')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

# 검증 데이터셋에 대한 예측 시각화 실행
plot_predictions(model_eval, val_loader, scaler_y)

# 학습 및 검증 손실 그래프를 그리는 함수
def plot_loss(train_losses, val_losses):
    # (함수 내용 생략)
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss (V2 Reinforced)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid()
    plt.show()

# 손실 그래프 시각화 실행
plot_loss(train_losses, val_losses)

print("\n--- 모델 성능 추가 분석 시작 ---\n")

def evaluate_and_analyze(model, val_loader, scaler_y, X_val, y_val, symbol_val, date_val):
    model.eval()
    preds_scaled, targets_scaled = [], []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds_scaled.extend(outputs.cpu().numpy())
            targets_scaled.extend(targets.cpu().numpy())

    preds = scaler_y.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    targets = scaler_y.inverse_transform(np.array(targets_scaled).reshape(-1, 1)).flatten()

    # 3. 예측 결과를 DataFrame으로 정리 (첫 번째 정의)
    results_df = pd.DataFrame({
        'Date': date_val,
        'Symbol': symbol_val,
        'Actual': targets,
        'Predicted': preds
    })

    results_df['Squared_Error'] = (results_df['Actual'] - results_df['Predicted']) ** 2
    # ⭐ 오류 해결: Absolute_Error 컬럼을 여기서 생성합니다.
    results_df['Absolute_Error'] = np.abs(results_df['Actual'] - results_df['Predicted'])

    # [⭐ 제거된 부분: 아래 3줄은 불필요한 중복 정의이므로 제거해야 합니다.]
    # results_df = pd.DataFrame({'Date': date_val, 'Symbol': symbol_val, 'Actual': targets, 'Predicted': preds})
    # results_df['Squared_Error'] = (results_df['Actual'] - results_df['Predicted']) ** 2

    # 4. 종목별 RMSE 계산 및 출력
    mse_per_symbol = results_df.groupby('Symbol')['Squared_Error'].mean().sort_values(ascending=False)
    rmse_per_symbol = np.sqrt(mse_per_symbol)

    print("## 📈 종목별 RMSE (Root Mean Squared Error) - 성능 분석 (Validation Set - V2)")
    print(rmse_per_symbol.to_string(float_format='%.4f'))
    print("\n" + "="*50 + "\n")

    # 5. 가장 잘 예측된 시점 찾기 (Absolute_Error 컬럼 사용)
    best_sample = results_df.sort_values('Absolute_Error').iloc[0]
    best_date = best_sample['Date']
    best_symbol = best_sample['Symbol']
    symbol_data = data[data['Symbol'] == best_symbol].sort_values('Date').reset_index(drop=True)
    end_index = symbol_data[symbol_data['Date'] == best_date].index[0]
    start_index = end_index - window_size
    best_input_data = symbol_data.iloc[start_index : end_index]

    print(f"## ⭐ 가장 잘 예측된 시점의 상세 분석 (절대 오차 최소)")
    print(f" - 예측 시점: {best_date}")
    print(f" - 종목: {best_symbol}")
    print(f" - 실제 다음날 종가: {best_sample['Actual']:.2f}")
    print(f" - 예측된 다음날 종가: {best_sample['Predicted']:.2f}")
    print(f" - 절대 오차: {best_sample['Absolute_Error']:.4f}\n")
    print(f"## 📜 입력 데이터 ({window_size}일 시퀀스)")
    print(best_input_data[['Date'] + FEATURES].to_string(index=False, float_format='%.4f'))

    return results_df

# 분석 실행
results_df = evaluate_and_analyze(model_eval, val_loader, scaler_y, X_val, y_val, symbol_val, date_val)

# --- 개별 종목 예측 시각화 (validations set) ---
def plot_single_symbol(results_df, symbol, suffix="Validation Set V2"):
    symbol_df = results_df[results_df['Symbol'] == symbol].reset_index(drop=True)
    if symbol_df.empty:
        print(f"\n경고: 종목 {symbol}에 대한 데이터가 검증 셋에 없습니다.")
        return

    plt.figure(figsize=(14, 7))
    plt.plot(symbol_df['Date'], symbol_df['Actual'], label=f'{symbol} Actual Close', linewidth=2)
    plt.plot(symbol_df['Date'], symbol_df['Predicted'], label=f'{symbol} Predicted Close', linewidth=2, alpha=0.8, linestyle='--')

    plt.title(f'{symbol}: Actual vs Predicted Next Day Close ({suffix})')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# RMSE가 가장 높았던 (가장 예측이 어려웠던) 종목 시각화
worst_symbol = results_df.groupby('Symbol')['Squared_Error'].mean().idxmax()
print(f"\n--- 가장 예측이 어려웠던 종목 시각화: {worst_symbol} ---\n")
plot_single_symbol(results_df, worst_symbol)

# RMSE가 가장 낮았던 (가장 예측이 쉬웠던) 종목 시각화
best_overall_symbol = results_df.groupby('Symbol')['Squared_Error'].mean().idxmin()
print(f"\n--- 가장 예측이 쉬웠던 종목 시각화: {best_overall_symbol} ---\n")
plot_single_symbol(results_df, best_overall_symbol)

# ----------------------------------------------------
# ⭐⭐ 2025년 1월 예측 결과 분석 추가 ⭐⭐
# ----------------------------------------------------

def get_predictions(model, data_loader, scaler_y):
    model.eval()
    preds_scaled, targets_scaled = [], []
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds_scaled.extend(outputs.cpu().numpy())
            targets_scaled.extend(targets.cpu().numpy())

    preds_scaled = np.array(preds_scaled)
    targets_scaled = np.array(targets_scaled)

    preds_price = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    targets_price = scaler_y.inverse_transform(targets_scaled.reshape(-1, 1)).flatten()

    return preds_price, targets_price

# 1. 모델 로드 및 예측 수행 (전체 시퀀스)
preds_price_all, targets_price_all = get_predictions(model_eval, pred_all_loader, scaler_y)

# 2. 예측 결과 DataFrame 재정의
pred_results_df_all = pd.DataFrame({
    'Date': date_pred_all,
    'Symbol': symbol_pred_all,
    'Actual_Next_Close': targets_price_all,
    'Predicted_Next_Close': preds_price_all
}).sort_values(['Symbol', 'Date']).reset_index(drop=True)

if pred_results_df_all.empty:
    print("\n[재시도 실패] 예측된 시퀀스가 없어 결과를 분석할 수 없습니다. 데이터 파일을 확인해주세요.")
    sys.exit()

print("\n" + "="*50)
print("## ✅ 전체 유효 예측 결과 분석")
print(f"총 유효 예측 샘플 수: {len(pred_results_df_all)}개")
print(f"최초 예측일: {pred_results_df_all['Date'].min()}")
print(f"최종 예측일: {pred_results_df_all['Date'].max()}")
print("="*50 + "\n")

# 3. 2025년 1월 예측 결과 필터링
start_date_pred_target = pd.to_datetime('2025-01-01')
end_date_pred_target = pd.to_datetime('2025-01-31')

pred_results_df_jan = pred_results_df_all[
    (pred_results_df_all['Date'] >= start_date_pred_target) &
    (pred_results_df_all['Date'] <= end_date_pred_target)
].sort_values(['Symbol', 'Date']).reset_index(drop=True)

if pred_results_df_jan.empty:
    print("\n[결과 없음] 2025년 1월 예측에 해당하는 유효한 데이터가 없습니다.")
    sys.exit()

print("="*50)
print("## ✅ 2025년 1월 예측 결과 (V2 Reinforced)")
print(f"총 유효 예측 샘플 수: {len(pred_results_df_jan)}개")
print(pred_results_df_jan.head().to_string(float_format='%.4f'))
print("="*50 + "\n")

# 4. 2025년 1월 예측 종가 vs 실제 종가 시각화 및 RMSE 계산
def plot_single_symbol_jan(results_df, symbol):
    symbol_df = results_df[results_df['Symbol'] == symbol].reset_index(drop=True)
    if symbol_df.empty:
        return

    plt.figure(figsize=(14, 7))
    plt.plot(symbol_df['Date'], symbol_df['Actual_Next_Close'], label=f'{symbol} Actual Next Close', linewidth=2)
    plt.plot(symbol_df['Date'], symbol_df['Predicted_Next_Close'], label=f'{symbol} Predicted Next Close', linewidth=2, alpha=0.8, linestyle='--')

    mse = ((symbol_df['Actual_Next_Close'] - symbol_df['Predicted_Next_Close']) ** 2).mean()
    rmse = np.sqrt(mse)

    plt.title(f'January 2025 Forecast (W=10) - {symbol}: V2 Reinforced (RMSE: {rmse:.4f})')
    plt.xlabel('Date')
    plt.ylabel('Next Day Close Price')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 2025년 1월 예측 결과에 대해 시각화 실행
predicted_symbols_jan = pred_results_df_jan['Symbol'].unique()
print(f"### 📈 2025년 1월 예측 시각화 시작 ({len(predicted_symbols_jan)} 종목)")

for symbol in predicted_symbols_jan:
    plot_single_symbol_jan(pred_results_df_jan, symbol)

# 전체 종목에 대한 RMSE 계산
mse_pred_per_symbol_jan = pred_results_df_jan.groupby('Symbol').apply(
    lambda x: ((x['Actual_Next_Close'] - x['Predicted_Next_Close']) ** 2).mean()
)
rmse_pred_per_symbol_jan = np.sqrt(mse_pred_per_symbol_jan).sort_values(ascending=False)

print("\n## 💰 2025년 1월 종목별 예측 RMSE (V2 Reinforced)")
print(rmse_pred_per_symbol_jan.to_string(float_format='%.4f'))

!jupyter nbconvert --to script /content/drive/MyDrive/Colab Notebooks/ver6.ipynb



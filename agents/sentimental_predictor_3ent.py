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
import sys 
from datetime import timedelta

# --- 데이터 불러오기 및 전처리 (학습 데이터) ---
stock_df = pd.read_csv('C:\\Users\\jinfo\\Desktop\\Programming\\capstone_project\\stock_data.csv') # 주가 데이터 불러오기
news_df = pd.read_csv('C:\\Users\\jinfo\\Desktop\\Programming\\capstone_project\\news_data.csv') # 뉴스 데이터 불러오기
stock = stock_df.copy() # 원본 보존을 위해 복사
news = news_df.copy()

# 뉴스 데이터 전처리
news['summary'] = news['summary'].fillna('') # 요약(summary) 컬럼의 결측값(NaN)을 빈 문자열로 채움
news = news.dropna(subset=['date']) # 날짜(date) 컬럼에 결측값이 있는 행 제거
news['date'] = pd.to_datetime(news['date']) # 날짜 컬럼을 datetime 객체로 변환

# FinBERT 모델 및 토크나이저 로드 (감성 분석용)
tokenizer = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone') 
finbert = AutoModelForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone') 
news['text'] = news['title'] + ' ' + news['summary'] # 제목과 요약을 합쳐 감성 분석에 사용할 텍스트 생성

# FinBERT를 사용하여 감성 점수를 계산하는 함수
def finbert_sentiment_scores(texts, batch_size=16):
    finbert.eval() # 모델을 평가 모드로 설정
    scores = []
    with torch.no_grad(): # 기울기 계산을 비활성화 (메모리 절약 및 속도 향상)
        for i in range(0, len(texts), batch_size): # 배치 단위로 텍스트 처리
            batch_texts = texts[i:i+batch_size]
            # 텍스트를 토크나이징 및 PyTorch 텐서로 변환
            inputs = tokenizer(list(batch_texts), return_tensors='pt', padding=True, truncation=True, max_length=512)
            outputs = finbert(**inputs) # 모델 예측
            # 로짓(logits)을 소프트맥스(softmax)를 통해 확률로 변환하고 NumPy 배열로 저장
            scores.extend(torch.softmax(outputs.logits, dim=1).cpu().numpy())
    return np.array(scores)


# 뉴스 데이터에 감성 점수(긍정/부정/중립 확률) 추가
news[['prob_positive', 'prob_negative', 'prob_neutral']] = finbert_sentiment_scores(news['text'].values)
news['date'] = pd.to_datetime(news['date']).dt.normalize() # 날짜의 시/분/초 제거 (일 단위로 통일)

# 일별/종목별 평균 감성 점수 및 뉴스 기사 수 계산
daily_sentiments = (news.groupby(['date','ticker'])
    .agg(prob_positive=('prob_positive','mean'), # 일별 평균 긍정 확률
         prob_negative=('prob_negative','mean'), # 일별 평균 부정 확률
         prob_neutral=('prob_neutral','mean'), # 일별 평균 중립 확률
         n_news=('title','count')).reset_index()) # 일별 뉴스 기사 수

# 기술적 분석 지표(Feature)를 추가하는 함수 (현재 모델에는 사용되지 않지만, 예측 시퀀스 완성을 위해 유지)
def add_tech_features(df):
    df['ma5']     = df['Close'].rolling(window=5).mean() 
    df['ma10']     = df['Close'].rolling(window=10).mean() 
    df['vol5']     = df['Volume'].rolling(window=5).mean() if 'Volume' in df.columns else 0
    df['ret']     = df['Close'].pct_change() # 일별 수익률 (전일 대비 변화율)
    df['ret_next']     = df['ret'].shift(-1) 
    return df

# 주가 데이터에 기술적 지표 추가 및 전처리
stock['Date'] = pd.to_datetime(stock['Date']).dt.normalize() # 날짜 형식 통일
# 종목별(Symbol)로 정렬 후 기술적 지표 계산
stock = stock.sort_values(['Symbol','Date']).groupby('Symbol', group_keys=False).apply(add_tech_features)
stock['next_close'] = stock.groupby('Symbol')['Close'].shift(-1) # 다음 날 종가를 예측 목표(Target)로 설정
# 수익률과 타겟값의 결측치 제거
stock = stock.dropna(subset=['ret','next_close']) 

# 주가 데이터와 일별 감성 데이터를 병합
data = pd.merge(stock, daily_sentiments, how='left',
    left_on=['Date','Symbol'], right_on=['date','ticker'])
# 병합 후 감성 관련 데이터 결측치(뉴스가 없던 날)는 0.0으로 채움
for c in ['prob_positive','prob_negative','prob_neutral','n_news']:
    if c in data: data[c] = data[c].fillna(0.0)

# LSTM 모델의 입력으로 사용할 특성(Feature) 리스트 정의
FEATURES = ['prob_positive','prob_negative','prob_neutral','n_news','ret','Close'] 

# 시계열 데이터(시퀀스)를 생성하는 함수
def create_sequences(df, window_size=10):
    df = df.sort_values('Date')
    arr = df[FEATURES].values # 특성 데이터 (입력 X)
    tgt = df['next_close'].values # 다음 날 종가 (출력 y)
    seqs, tgts = [], []
    for i in range(len(df) - window_size):
        seqs.append(arr[i:i+window_size]) # window_size 기간의 입력 시퀀스
        tgts.append(tgt[i+window_size]) # 시퀀스 다음 날의 종가
    return np.array(seqs), np.array(tgts)

# 전체 데이터셋을 시퀀스 형태로 준비하는 함수
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

window_size = 10 # 입력 시퀀스 길이 설정 (여기서는 10일)

# 학습 및 검증 데이터 기간 분할
train_df = data[(data['Date']>=pd.to_datetime('2024-01-01')) & (data['Date']<=pd.to_datetime('2024-10-31'))]
val_df = data[(data['Date']>=pd.to_datetime('2024-11-01')) & (data['Date']<=pd.to_datetime('2024-12-31'))]

# 학습 및 검증 데이터셋 준비
X_train, y_train, symbol_train, date_train = prepare_dataset(train_df, window_size)
X_val, y_val, symbol_val, date_val = prepare_dataset(val_df, window_size)

# 데이터 표준화 함수
def standardize(X, y):
    nsamples, nwin, nfeat = X.shape
    # 입력(X) 데이터를 평탄화하여 스케일러 학습 후 변환
    scaler_x = StandardScaler().fit(X.reshape(-1, nfeat))
    X_scaled = scaler_x.transform(X.reshape(-1, nfeat)).reshape(nsamples, nwin, nfeat)
    # 타겟(y) 데이터도 스케일러 학습 후 변환
    scaler_y = StandardScaler().fit(y.reshape(-1, 1))
    y_scaled = scaler_y.transform(y.reshape(-1, 1)).flatten()
    return X_scaled, y_scaled, scaler_x, scaler_y

# 학습 및 검증 데이터 표준화 및 스케일러 저장
X_train_scaled, y_train_scaled, scaler_x, scaler_y = standardize(X_train, y_train)
# 주의: 검증 데이터는 학습 데이터의 스케일러(scaler_x, scaler_y)를 사용해야 정확하지만 원본 코드와 동일하게 독립적으로 표준화 -> 이게 뭔 소리일까..
X_val_scaled, y_val_scaled, _, _ = standardize(X_val, y_val) 

# PyTorch Dataset 클래스 정의
class StockDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X, self.y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X) 
    def __getitem__(self, idx): return self.X[idx], self.y[idx] 

# DataLoader를 사용하여 데이터셋 배치 처리 준비
train_loader = torch.utils.data.DataLoader(StockDataset(X_train_scaled, y_train_scaled), batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(StockDataset(X_val_scaled, y_val_scaled), batch_size=32, shuffle=False)

# --- LSTM 모델 정의 ---
class StockSentimentLSTM(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=1, input_size=len(FEATURES)): 
        super().__init__()
        # input_size는 FEATURES 리스트의 길이(6)
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, 1) # 예측 종가(1차원) 출력
    def forward(self, x):
        out, _ = self.lstm(x) # LSTM 순전파
        # 마지막 시점의 출력만 사용하여 FC 레이어에 전달
        return self.fc(out[:, -1, :]).squeeze() 

# 모델 학습 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # GPU 또는 CPU 선택 -> 일단 CPU
model = StockSentimentLSTM().to(device) 
criterion = nn.MSELoss() # 손실 함수: 평균 제곱 오차 (MSE)
optimizer = optim.Adam(model.parameters(), lr=3e-4) # 최적화 함수: Adam

# --- 모델 학습 함수 ---
def train_model(train_loader, val_loader, model, criterion, optimizer, device, epochs=20):
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        t1, train_loss = time.time(), 0
        model.train() # 학습 모드
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
        model.eval() # 평가 모드
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

# --- 모델 및 스케일러 저장 ---
torch.save(model.state_dict(), "model_lstm_stocksentiment.pt") # 학습된 모델의 가중치 저장
joblib.dump(scaler_x, "scaler_x.pkl") # 입력 데이터 스케일러 저장
joblib.dump(scaler_y, "scaler_y.pkl") # 타겟 데이터 스케일러 저장


print("모델과 스케일러 저장 완료")

# --- 모델/스케일러 불러오기 및 예측 (validation set) ---
model_eval = StockSentimentLSTM(hidden_dim=64, num_layers=1, input_size=len(FEATURES)) 
model_eval.load_state_dict(torch.load("model_lstm_stocksentiment.pt")) # 저장된 가중치 로드
model_eval.eval() 
model_eval.to(device) 
scaler_x = joblib.load("scaler_x.pkl") 
scaler_y = joblib.load("scaler_y.pkl") 

# 예측 결과를 시각화하는 함수
def plot_predictions(model, val_loader, scaler_y):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for inputs, y_true in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds.extend(outputs.cpu().numpy())
            targets.extend(y_true.numpy())

    # 원래 스케일로 역변환
    preds = scaler_y.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
    targets = scaler_y.inverse_transform(np.array(targets).reshape(-1,1)).flatten()

    plt.figure(figsize=(12, 6))
    plt.plot(targets, label='Actual Next Close', linewidth=2)
    plt.plot(preds, label='Predicted Next Close', linewidth=2, alpha=0.8)
    # 제목에서 no_tech 제거
    plt.title('Actual vs Predicted Next Day Close (Validation set)')
    plt.xlabel('Sample Index (Time-ordered)')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

# 검증 데이터셋에 대한 예측 시각화 실행
plot_predictions(model_eval, val_loader, scaler_y)

# 학습 및 검증 손실 그래프를 그리는 함수
def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid()
    plt.show()

# 손실 그래프 시각화 실행
plot_loss(train_losses, val_losses)

print("\n--- 모델 성능 추가 분석 시작 ---\n")

def evaluate_and_analyze(model, val_loader, scaler_y, X_val, y_val, symbol_val, date_val):
    """
    검증 데이터셋에 대한 예측을 수행하고, 실제값과 예측값을 원래 스케일로 되돌린 후,
    종목별 MSE를 계산하고 가장 성능이 좋았던 예측 시점의 데이터를 반환
    """
    model.eval()
    preds_scaled, targets_scaled = [], []
    
    # 1. 스케일링된 예측값 및 실제값 추출
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds_scaled.extend(outputs.cpu().numpy())
            targets_scaled.extend(targets.cpu().numpy())
            
    preds_scaled = np.array(preds_scaled)
    targets_scaled = np.array(targets_scaled)
    
    # 2. 원래 가격 스케일로 역변환 (Inverse Transform)
    preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    targets = scaler_y.inverse_transform(targets_scaled.reshape(-1, 1)).flatten()

    # 3. 예측 결과를 DataFrame으로 정리
    results_df = pd.DataFrame({
        'Date': date_val,
        'Symbol': symbol_val,
        'Actual': targets,
        'Predicted': preds
    })
    
    results_df['Squared_Error'] = (results_df['Actual'] - results_df['Predicted']) ** 2
    results_df['Absolute_Error'] = np.abs(results_df['Actual'] - results_df['Predicted'])

    # 4. 종목별 RMSE 계산 및 출력
    mse_per_symbol = results_df.groupby('Symbol')['Squared_Error'].mean().sort_values(ascending=False)
    rmse_per_symbol = np.sqrt(mse_per_symbol)
    
    print("## 📈 종목별 RMSE (Root Mean Squared Error) - 성능 분석")
    print(rmse_per_symbol.to_string(float_format='%.4f'))
    print("\n" + "="*50 + "\n")

    # 5. 전체 데이터에서 가장 잘 예측된(오차가 작은) 샘플 찾기
    best_sample = results_df.sort_values('Absolute_Error').iloc[0]
    
    best_date = best_sample['Date']
    best_symbol = best_sample['Symbol']
    
    # 6. 가장 잘 예측된 시점의 입력 데이터 추출
    
    # 해당 종목의 전체 시계열 데이터
    symbol_data = data[data['Symbol'] == best_symbol].sort_values('Date').reset_index(drop=True)
    
    # 예측 시점의 인덱스를 찾음
    end_index = symbol_data[symbol_data['Date'] == best_date].index[0]
    start_index = end_index - window_size
    
    # 해당 시점의 입력 데이터 (시퀀스) 추출
    best_input_data = symbol_data.iloc[start_index : end_index]
    
    print(f"## ⭐ 가장 잘 예측된 시점의 상세 분석 (절대 오차 최소)")
    print(f" - 예측 시점: {best_date.strftime('%Y-%m-%d')}")
    print(f" - 종목: {best_symbol}")
    print(f" - 실제 다음날 종가: {best_sample['Actual']:.2f}")
    print(f" - 예측된 다음날 종가: {best_sample['Predicted']:.2f}")
    print(f" - 절대 오차: {best_sample['Absolute_Error']:.4f}\n")
    
    print(f"## 📜 입력 데이터 ({window_size}일 시퀀스) - {best_symbol} ({best_input_data['Date'].min().strftime('%Y-%m-%d')} ~ {best_input_data['Date'].max().strftime('%Y-%m-%d')})")
    print(best_input_data[['Date'] + FEATURES].to_string(index=False, float_format='%.4f'))

    return results_df

# 분석 실행
results_df = evaluate_and_analyze(model_eval, val_loader, scaler_y, X_val, y_val, symbol_val, date_val)

# --- 개별 종목 예측 시각화 (validations set) ---

def plot_single_symbol(results_df, symbol):
    """특정 종목에 대한 실제값과 예측값을 시각화"""
    symbol_df = results_df[results_df['Symbol'] == symbol].reset_index(drop=True)
    if symbol_df.empty:
        print(f"\n경고: 종목 {symbol}에 대한 데이터가 검증 셋에 없습니다.")
        return

    plt.figure(figsize=(14, 7))
    plt.plot(symbol_df['Date'], symbol_df['Actual'], label=f'{symbol} Actual Close', linewidth=2)
    plt.plot(symbol_df['Date'], symbol_df['Predicted'], label=f'{symbol} Predicted Close', linewidth=2, alpha=0.8, linestyle='--')
    
    plt.title(f'{symbol}: Actual vs Predicted Next Day Close')
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
# 1. 모델 로드 및 예측 수행 (3,735개 시퀀스 전체) - 재실행 필요 없음.
#    이전 코드에서 pred_results_df_all에 필요한 변수들이 이미 생성되었다고 가정합니다.
# ----------------------------------------------------

# 2. 예측 결과 DataFrame 재정의 (전체 유효 예측 결과)
#    (이 부분이 2025년 1월 데이터까지 포함된 전체 예측 결과를 담고 있습니다.)
pred_results_df_all = pd.DataFrame({
    'Date': date_pred,
    'Symbol': symbol_pred,
    'Actual_Next_Close': targets_price,
    'Predicted_Next_Close': preds_price
}).sort_values(['Symbol', 'Date']).reset_index(drop=True)

if pred_results_df_all.empty:
    print("\n[재시도 실패] 예측된 시퀀스가 없어 결과를 분석할 수 없습니다. 데이터 파일을 확인해주세요.")
    sys.exit()

print("\n" + "="*50)
print("## ✅ 전체 유효 예측 결과 분석")
print(f"총 유효 예측 샘플 수: {len(pred_results_df_all)}개")
print(f"최초 예측일: {pred_results_df_all['Date'].min().strftime('%Y-%m-%d')}")
print(f"최종 예측일: {pred_results_df_all['Date'].max().strftime('%Y-%m-%d')}")
print("="*50 + "\n")

# ----------------------------------------------------
# 3. 2025년 1월 예측 결과 필터링
# ----------------------------------------------------
start_date_pred_target = pd.to_datetime('2025-01-01')
end_date_pred_target = pd.to_datetime('2025-01-31')

pred_results_df_jan = pred_results_df_all[
    (pred_results_df_all['Date'] >= start_date_pred_target) & 
    (pred_results_df_all['Date'] <= end_date_pred_target)
].sort_values(['Symbol', 'Date']).reset_index(drop=True)

if pred_results_df_jan.empty:
    print("\n[결과 없음] 2025년 1월 예측에 해당하는 유효한 데이터가 없습니다. 데이터 수집 단계가 2025년 1월 31일까지 완료되었는지 확인해주세요.")
    sys.exit()

print("="*50)
print("## ✅ 2025년 1월 예측 결과")
print(f"총 유효 예측 샘플 수: {len(pred_results_df_jan)}개")
print(f"최초 예측일: {pred_results_df_jan['Date'].min().strftime('%Y-%m-%d')}")
print(f"최종 예측일: {pred_results_df_jan['Date'].max().strftime('%Y-%m-%d')}")
print(pred_results_df_jan.head().to_string(float_format='%.4f'))
print("="*50 + "\n")

# ----------------------------------------------------
# 4. 2025년 1월 예측 종가 vs 실제 종가 시각화 및 RMSE 계산
# ----------------------------------------------------

def plot_single_symbol_jan(results_df, symbol):
    """2025년 1월 특정 종목에 대한 실제값과 예측값을 시각화"""
    symbol_df = results_df[results_df['Symbol'] == symbol].reset_index(drop=True)
    if symbol_df.empty:
        return

    plt.figure(figsize=(14, 7))
    plt.plot(symbol_df['Date'], symbol_df['Actual_Next_Close'], label=f'{symbol} Actual Next Close', linewidth=2)
    plt.plot(symbol_df['Date'], symbol_df['Predicted_Next_Close'], label=f'{symbol} Predicted Next Close', linewidth=2, alpha=0.8, linestyle='--')
    
    # RMSE 계산 및 제목에 추가
    mse = ((symbol_df['Actual_Next_Close'] - symbol_df['Predicted_Next_Close']) ** 2).mean()
    rmse = np.sqrt(mse)
    
    plt.title(f'January 2025 Forecast (W=10) - {symbol}: Actual vs Predicted Next Day Close (RMSE: {rmse:.4f})')
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

print("\n## 💰 2025년 1월 종목별 예측 RMSE")
print(rmse_pred_per_symbol_jan.to_string(float_format='%.4f'))
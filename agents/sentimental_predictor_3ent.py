import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sklearn.model_selection import train_test_split

# 1. 뉴스 데이터 로드 및 전처리
news = pd.read_csv('news_data.csv')
news['date'] = pd.to_datetime(news['date']).dt.date
news['text'] = news['title'].fillna('') + ' ' + news['summary'].fillna('')

# 2. FinBERT 토크나이저 및 모델 로드
tokenizer = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')
model_bert = AutoModel.from_pretrained('yiyanghkust/finbert-tone')

# 3. FinBERT 임베딩 함수 정의 (batch 처리 기능 추가)
def finbert_sentiment_score(texts, batch_size=16):
    embeddings = []
    model_bert.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
            outputs = model_bert(**inputs)
            batch_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.extend(batch_emb)
    return np.array(embeddings)

# 4. 뉴스별 임베딩 생성 (속도 개선: batch 처리)
news_embeddings = finbert_sentiment_score(news['text'].tolist())
news['embedding'] = list(news_embeddings)

# 5. 일별, 종목별 임베딩 평균 계산 (일별 대표 임베딩)
news_date_group = news.groupby(['date', 'related'])['embedding'] \
                     .apply(lambda x: np.mean(np.stack(x), axis=0)) \
                     .reset_index() \
                     .rename(columns={'embedding':'daily_sentiment_embedding'})

# 6. 주가 데이터 로드 및 전처리
stock = pd.read_csv('stock_data.csv')
stock['Date'] = pd.to_datetime(stock['Date']).dt.date
stock = stock.sort_values(by=['Symbol', 'Date'])
stock['next_close'] = stock.groupby('Symbol')['Close'].shift(-1)
stock['log_return'] = np.log(stock['next_close'] / stock['Close'])
stock = stock.dropna(subset=['log_return'])

# 7. 뉴스 임베딩 데이터와 주가 데이터 병합 (내부 조인)
data = pd.merge(news_date_group, stock, how='inner', left_on=['date','related'], right_on=['Date','Symbol'])

# 8. 학습 및 검증 데이터 분할 - 날짜 대신 시계열 검증 고려 가능
train_start = pd.to_datetime('2023-09-01').date()
train_end = pd.to_datetime('2024-11-30').date()
val_start = pd.to_datetime('2024-12-01').date()
val_end = pd.to_datetime('2024-12-27').date()

train_df = data[(data['date'] >= train_start) & (data['date'] <= train_end)]
val_df = data[(data['date'] >= val_start) & (data['date'] <= val_end)]

# 9. PyTorch 데이터셋 정의 (배치 시 임베딩 크기 검증 추가)
class StockNewsDataset(Dataset):
    def __init__(self, df):
        self.xs = list(df['daily_sentiment_embedding'])
        self.ys = list(df['log_return'])
    def __len__(self):
        return len(self.ys)
    def __getitem__(self, idx):
        return torch.tensor(self.xs[idx], dtype=torch.float32), torch.tensor(self.ys[idx], dtype=torch.float32)

train_dataset = StockNewsDataset(train_df)
val_dataset = StockNewsDataset(val_df)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=32, drop_last=False)

# 10. MLP 모델 정의 (3개 은닉층, Dropout 및 BatchNorm 추가로 과적합 완화)
class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLPRegressor(input_dim=768).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)  # weight_decay로 정규화 추가
criterion = nn.MSELoss()

# 11. 학습 루프 (에포크마다 검증 수행, early stopping 구현 가능)
epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.6f}")

    # 검증 루프
    model.eval()
    val_losses = []
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            pred_val = model(x_val)
            val_loss = criterion(pred_val, y_val)
            val_losses.append(val_loss.item())
    print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {np.mean(val_losses):.6f}")

# 12. 최종 평가 및 예측 결과 산출
model.eval()
val_preds, val_trues = [], []
with torch.no_grad():
    for x_val, y_val in val_loader:
        x_val, y_val = x_val.to(device), y_val.to(device)
        outputs = model(x_val)
        val_preds.extend(outputs.cpu().numpy())
        val_trues.extend(y_val.cpu().numpy())

val_preds = np.array(val_preds)
val_trues = np.array(val_trues)

base_close = val_df['Close'].values[:len(val_preds)]
predicted_close = base_close * np.exp(val_preds)
actual_close = base_close * np.exp(val_trues)

mape_price = np.mean(np.abs((predicted_close - actual_close) / base_close)) * 100
rmse_price = np.sqrt(np.mean((predicted_close - actual_close) ** 2))
correct_direction = np.sign(predicted_close - base_close) == np.sign(actual_close - base_close)
direction_acc = 100 * np.mean(correct_direction)

print("\n[테스트 리포트]")
print(f"상승하락률 RMSE: {rmse_price:.8f}")
print(f"MAPE(다음 거래일 종가, %): {mape_price:.3f}")
print(f"방향 정확도(%): {direction_acc:.2f}")

# 13. 예측 결과 테이블 출력 (사용자 지정 날짜에 대하여)
dates_to_show = ['2024-12-20', '2024-12-23', '2024-12-24']
stock_korean = {"NVDA": "엔비디아", "AAPL": "애플", "MSFT": "마이크로소프트"}

output_rows = []
for symbol in val_df['Symbol'].unique():
    kor = stock_korean.get(symbol, symbol)
    subdf = val_df[(val_df['Symbol'] == symbol) & (val_df['date'].astype(str).isin(dates_to_show))].reset_index(drop=False)
    for date_str in dates_to_show:
        rows = subdf[subdf['date'].astype(str) == date_str]
        if rows.empty:
            continue
        idx = rows.index[0]
        기준일 = rows.loc[idx, 'date']
        next_idx = idx + 1 if idx + 1 < len(val_df) else idx
        예측대상일 = val_df.iloc[next_idx]['date'] if next_idx < len(val_df) else "N/A"
        기준종가 = base_close[idx]
        예측종가 = predicted_close[idx]
        실제종가 = actual_close[idx]
        오차 = 예측종가 - 실제종가
        오차율 = 100 * 오차 / 기준종가
        output_rows.append({
            "종목": f"{kor} ({symbol})",
            "기준일": 기준일,
            "예측대상일": 예측대상일,
            "기준일 종가": f"{기준종가:.2f}",
            "예측 종가": f"{예측종가:.2f}",
            "실제 종가": f"{실제종가:.2f}",
            "오차": f"{오차:+.2f}",
            "오차율(%)": f"{오차율:+.2f}%"
        })

result_df = pd.DataFrame(output_rows)
print(result_df)


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(style='whitegrid')

def plot_multiple_stocks(df, preds, trues, base_close, stock_map):
    symbols = list(df['Symbol'].unique())
    num_symbols = len(symbols)

    plt.figure(figsize=(14, 5 * num_symbols))  # 더 큰 높이로 조정

    for i, symbol in enumerate(symbols):
        eng_name = symbol
        subdf = df[df['Symbol'] == symbol].reset_index(drop=True)
        idxs = subdf.index

        dates = pd.to_datetime(subdf['date'])
        actual = base_close[idxs] * np.exp(trues[idxs])
        predicted = base_close[idxs] * np.exp(preds[idxs])
        base = base_close[idxs]

        plt.subplot(num_symbols, 1, i + 1)
        plt.plot(dates, actual, label='Actual Close', marker='o')
        plt.plot(dates, predicted, label='Predicted Close', marker='x')
        error_pct = 100 * (predicted - actual) / base
        plt.bar(dates, error_pct, alpha=0.3, label="Error Rate (%)", color='coral')

        plt.title(f'{eng_name}: Actual vs Predicted Close', fontsize=15)
        plt.xlabel('Date', fontsize=13)
        plt.ylabel('Price & Error (%)', fontsize=13)
        plt.legend()
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)  # 간격 충분히 넓힘
    plt.show()

plot_multiple_stocks(val_df.reset_index(drop=True), val_preds, val_trues, base_close, stock_korean)

# 모델 저장 경로 지정
model_save_path = 'mlp_stock_model.pt'

# 모델 저장 (state_dict)
torch.save(model.state_dict(), model_save_path)
print(f"모델이 '{model_save_path}' 파일로 저장되었습니다.")
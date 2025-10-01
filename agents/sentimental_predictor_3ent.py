import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import numpy as np

# 1. 뉴스 데이터 로드 및 전처리
news = pd.read_csv('C:\\Users\\jinfo\\Desktop\\Programming\\capstone_project\\agents\\news_data.csv')
news['date'] = pd.to_datetime(news['date']).dt.date
news['text'] = news['title'].fillna('') + ' ' + news['summary'].fillna('')
news['related'] = news['related'].fillna('')

# 2. 주가 데이터 로드 및 수익률 계산
stock = pd.read_csv('C:\\Users\\jinfo\\Desktop\\Programming\\capstone_project\\stock_data.csv')
stock['Date'] = pd.to_datetime(stock['Date']).dt.date
stock = stock.sort_values(by=['Symbol', 'Date'])

# next_close가 검증 기간에 있어야 하므로 충분한 기간 포함시킴
stock['next_close'] = stock.groupby('Symbol')['Close'].shift(-1)
stock['log_return'] = np.log(stock['next_close'] / stock['Close'])
stock = stock.dropna(subset=['log_return'])

# 3. 뉴스와 주가 매칭
data = pd.merge(news, stock, how='inner', left_on=['date', 'related'], right_on=['Date', 'Symbol'])

# 4. 날짜 컬럼 정리
date_col = 'date' if 'date' in data.columns else 'Date'
data[date_col] = pd.to_datetime(data[date_col]).dt.date

# 5. 날짜 기준 시계열 분할(예측 대상일도 고려해 하루 더 포함)
train_start = pd.to_datetime('2023-09-01').date()
train_end = pd.to_datetime('2024-12-31').date()
val_start = pd.to_datetime('2025-01-01').date()
val_end = pd.to_datetime('2025-03-31').date()
buffer_days = pd.Timedelta(1, unit='d')

train_df = data[(data[date_col] >= train_start) & (data[date_col] <= train_end)]
val_df = data[(data[date_col] >= val_start) & (data[date_col] <= val_end + buffer_days)]

# 6. FINBERT 임베딩 함수
tokenizer = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')
model = AutoModel.from_pretrained('yiyanghkust/finbert-tone')

def get_finbert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

train_df['embedding'] = train_df['text'].map(get_finbert_embedding)
val_df['embedding'] = val_df['text'].map(get_finbert_embedding)

# 7. Dataset & DataLoader
class StockNewsDataset(Dataset):
    def __init__(self, df):
        self.embeddings = list(df['embedding'])
        self.targets = list(df['log_return'])
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.float32)

train_dataset = StockNewsDataset(train_df)
val_dataset = StockNewsDataset(val_df)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# 8. MLP 모델
class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLPRegressor(input_dim=768).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 9. 학습
for epoch in range(10):
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
    print(f"Epoch {epoch+1}, Train Loss: {total_loss / len(train_loader)}")

# 10. 예측
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

# 11. 종가 복원
base_close = val_df['Close'].values[:len(val_preds)]
predicted_close = base_close * np.exp(val_preds)
actual_close = base_close * np.exp(val_trues)

# 12. 평가 지표
mape_price = np.mean(np.abs((predicted_close - actual_close) / base_close)) * 100
rmse_price = np.sqrt(np.mean((predicted_close - actual_close) ** 2))
correct_direction = np.sign(predicted_close - base_close) == np.sign(actual_close - base_close)
direction_acc = 100 * np.mean(correct_direction)

print("\n[테스트 리포트]")
print(f"상승하락률 RMSE: {rmse_price:.8f}")
print(f"MAPE(다음 거래일 종가, %): {mape_price:.3f}")
print(f"방향 정확도(%): {direction_acc:.2f}\n")

print("학습된 아티팩트 저장 시작..")
print("아티팩트 저장 완료. (폴더: model_artifacts)\n")
print("[다음 3거래일 예측 vs 실제]")

dates_to_show = ['2024-12-31', '2025-01-02', '2025-01-03']

stock_korean = {"NVDA": "엔비디아", "AAPL": "애플", "MSFT": "마이크로소프트"}

for symbol in val_df['Symbol'].unique():
    kor = stock_korean.get(symbol, symbol)
    print(f"\n종목: {kor} ({symbol})")

    subdf = val_df[(val_df['Symbol'] == symbol) & (val_df[date_col].astype(str).isin(dates_to_show))].reset_index(drop=True)

    for 기준_날짜 in dates_to_show:
        row = subdf[subdf[date_col].astype(str) == 기준_날짜]
        if row.empty:
            print(f"기준일={기준_날짜} 예측대상일=N/A 데이터 없음")
            continue
        idx = row.index[0]
        기준일 = row.iloc[0][date_col]
        next_idx = idx + 1 if idx + 1 < len(subdf) else idx
        예측대상일 = subdf.iloc[next_idx][date_col] if len(subdf) > 1 else "N/A"
        기준종가 = base_close[row.index[0]]
        예측종가 = predicted_close[row.index[0]]
        실제종가 = actual_close[row.index[0]]
        오차 = 예측종가 - 실제종가
        오차율 = 100 * 오차 / 기준종가
        sign = "+" if 오차율 >= 0 else ""
        print(
    f"기준일={기준일} 예측대상일={예측대상일} 기준일종가={기준종가:.2f} 예측={예측종가:.2f} 실제={실제종가:.2f} "
    f"오차={오차:+.2f} ({sign}{오차율:.2f}%)"
)

print(f"\n[요약] MAPE(다음 거래일 종가, %): {mape_price:.2f}%")
#!/usr/bin/env python3
"""
간단한 matplotlib 테스트
"""

import matplotlib
matplotlib.use('Agg')  # 백엔드 설정
import matplotlib.pyplot as plt

print("🧪 간단한 matplotlib 테스트")

# 간단한 차트 생성
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([1, 2, 3], [1, 4, 2])
ax.set_title('테스트 차트')
ax.set_xlabel('X')
ax.set_ylabel('Y')

# 파일로 저장
plt.savefig('test_chart.png', dpi=150, bbox_inches='tight')
print("✅ 차트가 test_chart.png로 저장되었습니다.")

# 표 생성 테스트
fig, ax = plt.subplots(figsize=(10, 6))
table_data = [
    ['Round 1', '100.00 USD', '105.00 USD', '98.00 USD'],
    ['Round 2', '102.00 USD', '108.00 USD', '99.00 USD']
]
headers = ['라운드', 'Sentimental Agent', 'Technical Agent', 'Fundamental Agent']

table = ax.table(cellText=table_data, colLabels=headers, 
                cellLoc='center', loc='center', bbox=[0.1, 0.1, 0.8, 0.8])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

# 헤더 스타일링
for i in range(len(headers)):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

ax.set_title('투자의견 표 테스트', fontsize=16, fontweight='bold', pad=20)
ax.axis('off')

plt.savefig('test_table.png', dpi=150, bbox_inches='tight')
print("✅ 표가 test_table.png로 저장되었습니다.")

plt.close('all')
print("✅ 테스트 완료!")

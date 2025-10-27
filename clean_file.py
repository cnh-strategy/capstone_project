# clean_file.py
file_path = r'c:\Users\jinfo\Desktop\Programming\capstone_project\debate_ver3\agents\sentimental_agent.py'

# 파일 읽기
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 모든 비표준 공백 문자 제거
content = content.replace('\u00a0', ' ')  # non-breaking space
content = content.replace('\u2000', ' ')  # en quad
content = content.replace('\u2001', ' ')  # em quad
content = content.replace('\u2002', ' ')  # en space
content = content.replace('\u2003', ' ')  # em space
content = content.replace('\u202f', ' ')  # narrow no-break spacepyt
content = content.replace('\u205f', ' ')  # medium mathematical space

# 파일 쓰기
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ 파일 정리 완료!")

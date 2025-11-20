# SSH 키 GitHub 추가 가이드

## SSH 키 생성 완료! ✅

공개 키가 생성되었습니다. 이제 GitHub에 추가해야 합니다.

## 다음 단계

### 1. 공개 키 복사
공개 키는 이미 클립보드에 복사되었습니다. 아래 명령어로 다시 확인할 수 있습니다:
```bash
cat ~/.ssh/id_ed25519.pub
```

### 2. GitHub에 SSH 키 추가

1. **GitHub 웹사이트 접속**
   - https://github.com/settings/keys 로 이동
   - 또는 GitHub → Settings → SSH and GPG keys

2. **"New SSH key" 클릭**

3. **키 정보 입력**
   - Title: `MacBook Air` (또는 원하는 이름)
   - Key: 클립보드의 공개 키 붙여넣기
   - Key type: `Authentication Key`

4. **"Add SSH key" 클릭**

### 3. SSH 연결 테스트

GitHub에 키를 추가한 후:
```bash
ssh -T git@github.com
```

"Hi LukaKKS! You've successfully authenticated..." 메시지가 나오면 성공!

### 4. Push 시도

```bash
git push -u origin main
```

## 공개 키 (이미 클립보드에 복사됨)

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMlqLe54eAvaaBlG9zprZFmg406F7cqIXkcoAeOb4Vm4 157806956+LukaKKS@users.noreply.github.com
```


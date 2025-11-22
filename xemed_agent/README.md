# XEMED Agent

4-Agent 기반 환자 발화 분석 파이프라인

## 시스템 구조

전체 시스템은 다음 4개의 Agent로 구성된 직렬 파이프라인입니다:

1. **Emotion Agent** → 감정 신호 추출
2. **Clinical Agent** → 임상 신호 추출
3. **KG-Fusion Agent** → 지식그래프 기반 위험도 산출
4. **ESI Classifier** → 최종 ESI 레벨 분류

각 Agent는 독립적으로 작동하며, JSON 형식의 구조화된 출력을 다음 단계로 전달합니다.

---

## 1. Emotion Agent

### 기능

환자 발화(utterance)를 입력받아 다음을 추출:
- **Primary Emotion**: 기본 감정
- **Secondary Emotion + Score**: 보조 감정 및 강도 (0~1)
- **Subtle Emotion + Score**: 세밀 감정 및 강도 (0~1)
- **Masked Emotion**: 억압 감정
- **Emotion Risk Score**: 최종 감정 위험도 (0~1, LLM reasoning 기반)

### 특징

- LLM(GPT-4o-mini) 기반 감정 분석
- KG를 직접 사용하지 않음 (순수 LLM 기반)
- LangGraph를 사용한 워크플로우 관리
- JSON 형식의 구조화된 출력

### 사용 방법

```python
from emotion_agent import EmotionAgent

# Agent 초기화
agent = EmotionAgent()

# 단일 발화 처리
utterance = "I'm really worried about this pain in my chest."
result = agent.process(utterance)

print(result)
# {
#     "primary_emotion": "fear",
#     "secondary_emotion": "anxiety",
#     "secondary_score": 0.85,
#     "subtle_emotion": "uncertainty",
#     "subtle_score": 0.65,
#     "masked_emotion": "panic",
#     "emotion_risk_score": 0.75,
#     "reasoning": "...",
#     "error": None
# }

# 배치 처리
utterances = ["발화1", "발화2", "발화3"]
results = agent.process_batch(utterances)
```

### 출력 형식

```json
{
    "primary_emotion": "string",
    "secondary_emotion": "string or null",
    "secondary_score": 0.0-1.0 or null,
    "subtle_emotion": "string or null",
    "subtle_score": 0.0-1.0 or null,
    "masked_emotion": "string or null",
    "emotion_risk_score": 0.0-1.0,
    "reasoning": "string",
    "error": "string or null"
}
```

### 환경 설정

`.env` 파일에 다음을 설정:

```
OPENAI_API_KEY=your_api_key_here
```

---

## 설치

```bash
# 가상환경 활성화 (선택사항)
conda activate eclipse

# 패키지 설치
pip install -r requirements.txt
```

---

## 테스트

```bash
python emotion_agent.py
```

---

## 다음 단계

- [ ] Clinical Agent 구현
- [ ] KG-Fusion Agent 구현
- [ ] ESI Classifier 구현
- [ ] 전체 파이프라인 통합


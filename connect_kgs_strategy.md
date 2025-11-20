# 두 KG 연결 전략

## 📊 현재 두 KG 구조

### 1. SNOMED CT GPFP KG
- **노드**: `Concept` (의학 개념)
  - `id`: SNOMED CT ID (예: "140004")
  - `name`: Preferred Term (예: "Chronic pharyngitis")
  - `allTerms`: 모든 용어 리스트
- **관계**: `RELATES_TO` (IS_A, CAUSED_BY 등)

### 2. BCB KG
- **노드**: 
  - `Symptom` (UMLS 기반)
    - `cui`: UMLS CUI 코드 (예: "C0030193")
    - `name`: 증상 이름
    - `definition`: 정의
  - `Patient`, `Emotion`, `Utterance` 등
- **관계**: `HAS_SYMPTOM`, `EXPRESSES_EMOTION` 등

## 🔗 연결 방법 (우선순위 순)

### 방법 1: UMLS-SNOMED CT 매핑 파일 사용 (가장 정확)
**장점**: 
- 공식 매핑 데이터 사용
- 높은 정확도
- 자동화 가능

**단점**:
- UMLS Metathesaurus 라이선스 필요
- 매핑 파일 다운로드 필요

**구현**:
```cypher
// Symptom과 Concept를 매핑 테이블로 연결
MATCH (s:Symptom {cui: "C0030193"})
MATCH (c:Concept {id: "140004"})
MERGE (s)-[:MAPS_TO_SNOMED]->(c)
```

### 방법 2: 이름 기반 유사도 매칭 (실용적)
**장점**:
- 추가 데이터 불필요
- 즉시 구현 가능
- LLM으로 검증 가능

**단점**:
- 동의어 처리 필요
- 오매칭 가능성

**구현**:
- Symptom의 `name`과 Concept의 `name`/`allTerms` 비교
- 문자열 유사도 (Levenshtein, Jaccard)
- LLM으로 의미적 유사도 검증

### 방법 3: LLM 기반 의미 매칭 (가장 유연)
**장점**:
- 자연어 이해 활용
- 동의어 자동 처리
- 컨텍스트 고려

**단점**:
- API 비용
- 처리 시간

**구현**:
- LangGraph로 Symptom → Concept 매칭 워크플로우
- LLM이 Symptom 설명과 Concept 용어를 비교

### 방법 4: 하이브리드 접근 (권장) ⭐
**전략**:
1. **1차**: 이름 기반 정확 매칭 (exact match)
2. **2차**: 유사도 기반 매칭 (threshold > 0.9)
3. **3차**: LLM 검증 (불확실한 경우만)
4. **4차**: 수동 검토 큐 (매칭 실패)

## 🎯 권장 구현 방식

### Phase 1: 기본 연결 (이름 기반)
```python
# Symptom.name과 Concept.name/allTerms 매칭
# 정확 매칭 → MERGE 관계 생성
```

### Phase 2: 고급 연결 (LLM + 유사도)
```python
# LangGraph 워크플로우:
# 1. Symptom 추출
# 2. SNOMED CT Concept 검색 (이름/용어)
# 3. LLM으로 의미 검증
# 4. MAPS_TO_SNOMED 관계 생성
```

### Phase 3: 관계 확장
```cypher
// 환자가 가진 증상 → SNOMED CT 개념 → 관련 개념 탐색
MATCH (p:Patient)-[:HAS_SYMPTOM]->(s:Symptom)
      -[:MAPS_TO_SNOMED]->(c:Concept)
      -[:RELATES_TO]->(related:Concept)
RETURN p, s, c, related
```

## 📈 예상 효과

1. **의학적 맥락 확장**: 환자 증상 → SNOMED CT 개념 → 관련 질환/증상
2. **의학 지식 활용**: SNOMED CT의 계층 구조로 증상 분류
3. **질의 응답 향상**: "이 증상과 관련된 다른 질환은?" 같은 질문 가능
4. **임상 의사결정 지원**: 증상 패턴 → SNOMED CT 경로 → 진단 추론


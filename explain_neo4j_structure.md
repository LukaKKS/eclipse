# Neo4j SNOMED CT GPFP 구조 설명

## 📊 전체 구조 개요

Neo4j에는 **의학 개념(Concept)**들이 **노드(Node)**로 저장되고, 개념들 간의 **관계(Relationship)**가 **엣지(Edge)**로 연결되어 있습니다.

```
[Concept 노드] --[RELATES_TO 관계]--> [Concept 노드]
```

---

## 🏷️ Concept 노드 (의학 개념)

각 Concept 노드는 하나의 의학 개념을 나타냅니다.

### 노드 속성 (Properties)

| 속성명 | 설명 | 예시 |
|--------|------|------|
| `id` | SNOMED CT 개념 고유 ID | `"140004"` |
| `name` | 개념의 주요 이름 (Preferred Term) | `"Chronic pharyngitis"` |
| `allTerms` | 이 개념을 나타내는 모든 용어들의 리스트 | `["Chronic pharyngitis", "Chronic pharyngitis (disorder)"]` |
| `effectiveTime` | 개념이 유효한 날짜 | `"20020131"` (2002년 1월 31일) |
| `definitionStatusId` | 정의 상태 (완전 정의 vs 원시 정의) | `"900000000000073002"` |
| `moduleId` | 개념이 속한 모듈 ID | `"900000000000207008"` |

### 예시 노드

```
(:Concept {
  id: "140004",
  name: "Chronic pharyngitis",
  allTerms: ["Chronic pharyngitis", "Chronic pharyngitis (disorder)"],
  effectiveTime: "20020131",
  definitionStatusId: "900000000000073002",
  moduleId: "900000000000207008"
})
```

**의미**: "만성 인두염"이라는 의학 개념

---

## 🔗 RELATES_TO 관계 (개념 간 관계)

Concept 노드들을 연결하는 엣지입니다. 의학 개념들 간의 의미적 관계를 나타냅니다.

### 관계 속성 (Properties)

| 속성명 | 설명 | 예시 |
|--------|------|------|
| `id` | 관계의 고유 ID | `"2460597026"` |
| `typeId` | 관계 타입 ID (가장 중요!) | `"116680003"` (IS_A 관계) |
| `relationshipGroup` | 관계 그룹 번호 | `"0"` |
| `characteristicTypeId` | 관계 특성 타입 | `"900000000000011006"` |

### 주요 관계 타입 (typeId)

| typeId | 의미 | 설명 |
|--------|------|------|
| `116680003` | **IS_A** | "~는 ~의 일종이다" (계층 관계) |
| `47429007` | **CAUSED_BY** | "~는 ~에 의해 발생한다" (원인 관계) |
| `363698007` | **FINDING_SITE** | "~의 발견 부위" |
| `246075003` | **CAUSATIVE_AGENT** | "~의 원인 물질" |

---

## 📝 실제 예시 분석

### 예시 1: 만성 인두염 → 인두염

```
Chronic pharyngitis --[IS_A]--> Pharyngitis
```

**의미**: 
- "만성 인두염"은 "인두염"의 **하위 개념**입니다
- 즉, 만성 인두염은 인두염의 한 종류입니다

**구조**:
- 시작 노드: `Chronic pharyngitis` (id: 140004)
- 관계: `IS_A` (typeId: 116680003)
- 끝 노드: `Pharyngitis` (id: 405737000)

---

### 예시 2: 알코올성 치매 → 알코올 중독

```
Alcohol-induced persisting dementia --[CAUSED_BY]--> Alcohol problem drinking
```

**의미**:
- "알코올성 지속성 치매"는 "알코올 중독"에 **의해 발생**합니다
- 원인-결과 관계입니다

**구조**:
- 시작 노드: `Alcohol-induced persisting dementia` (id: 281004)
- 관계: `CAUSED_BY` (typeId: 47429007)
- 끝 노드: `Alcohol problem drinking` (id: 7200002)

---

### 예시 3: 급성 고막염 → 급성 중이염

```
Acute tympanitis --[IS_A]--> Acute otitis media
```

**의미**:
- "급성 고막염"은 "급성 중이염"의 **하위 개념**입니다
- 고막염은 중이염의 한 종류입니다

---

### 예시 4: 후두부 두통 → 두통

```
Occipital headache --[IS_A]--> Pain in head
```

**의미**:
- "후두부 두통"은 "두통"의 **하위 개념**입니다
- 더 구체적인 두통 유형입니다

---

## 🎯 그래프 구조의 장점

### 1. **계층적 탐색 가능**
```
만성 인두염
  └─ IS_A → 인두염
       └─ IS_A → 상부 호흡기 감염
            └─ IS_A → 감염
```

### 2. **다양한 관계 타입**
- **IS_A**: "~의 종류"
- **CAUSED_BY**: "~의 원인"
- **FINDING_SITE**: "~의 위치"
- **CAUSATIVE_AGENT**: "~의 원인 물질"

### 3. **유사 개념 검색**
같은 상위 개념을 가진 개념들을 쉽게 찾을 수 있습니다.

```cypher
// "인두염"의 모든 하위 개념 찾기
MATCH (parent:Concept {name: "Pharyngitis"})<-[:RELATES_TO {typeId: "116680003"}]-(child:Concept)
RETURN child.name
```

---

## 🔍 실제 사용 예시

### 질문: "만성 인두염과 관련된 더 넓은 개념은?"

```cypher
MATCH (c:Concept {id: "140004"})-[:RELATES_TO {typeId: "116680003"}]->(parent:Concept)
RETURN parent.name
```

**결과**: `Pharyngitis` (인두염)

### 질문: "인두염의 모든 종류는?"

```cypher
MATCH (parent:Concept {id: "405737000"})<-[:RELATES_TO {typeId: "116680003"}]-(child:Concept)
RETURN child.name
```

**결과**: `Chronic pharyngitis`, `Acute pharyngitis` 등

---

## 📊 데이터 규모

- **노드 수**: 약 4,253개의 Concept 노드
- **엣지 수**: 약 61,489개의 RELATES_TO 관계
- **용어 수**: 각 개념당 평균 2-3개의 용어 (allTerms)

---

## 💡 핵심 정리

1. **Concept = 의학 개념** (질병, 증상, 치료 등)
2. **RELATES_TO = 개념 간 관계** (IS_A, CAUSED_BY 등)
3. **계층 구조**: IS_A 관계로 상위-하위 개념 연결
4. **다양한 관계**: 원인, 위치, 물질 등 다양한 관계 타입
5. **용어 다양성**: 하나의 개념에 여러 용어 저장 (allTerms)

이 구조를 통해 의학 지식을 그래프로 표현하고, 복잡한 의학 개념 간의 관계를 탐색할 수 있습니다! 🎉


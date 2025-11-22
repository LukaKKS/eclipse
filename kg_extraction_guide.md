# 통합 KG에서 추출 가능한 정보 가이드

## 📊 KG 구조 개요

### 노드 타입
- **Concept**: 4,253개 (SNOMED CT GPFP 의학 개념)
- **Patient**: 989개 (BCB 환자)
- **Utterance**: 989개 (환자 발화)
- **Emotion**: 64개 (감정)
- **SubtleEmotion**: 379개 (세밀한 감정)
- **Explanation**: 1,419개 (감정 설명)
- **Section**: 20개 (임상 섹션: GENHX, CC 등)

### 관계 타입
- **RELATES_TO**: 2,656개 (Concept 간 관계)
- **MENTIONS_CONCEPT**: 783개 (Utterance → Concept 연결)
- **SPOKE**: 989개 (Patient → Utterance)
- **EXPRESSES_PRIMARY/SECONDARY/ANNOTATED**: 감정 표현
- **EXPRESSES_SUBTLE**: 세밀한 감정 표현
- **EXPLAINED_BY**: 감정 설명
- **HAS_SECTION**: 환자 섹션

---

## 🎯 추출 가능한 정보 카테고리

### 1. 환자 정보 추출

#### 1.1 환자 기본 정보
```cypher
// 특정 환자 정보
MATCH (p:Patient {id: "14"})
RETURN p
```

#### 1.2 환자의 모든 발화
```cypher
// 환자의 모든 발화 추출
MATCH (p:Patient {id: "14"})-[:SPOKE]->(u:Utterance)
RETURN u.text AS 발화
```

#### 1.3 환자의 감정 프로필
```cypher
// 환자가 표현한 모든 감정
MATCH (p:Patient {id: "14"})-[:EXPRESSES_PRIMARY|EXPRESSES_SECONDARY|EXPRESSES_ANNOTATED]->(e:Emotion)
RETURN e.label AS 감정, e.score AS 점수
```

#### 1.4 환자의 의학 개념 (증상/질환)
```cypher
// 환자가 언급한 모든 의학 개념
MATCH (p:Patient {id: "14"})-[:SPOKE]->(u:Utterance)
      -[:MENTIONS_CONCEPT]->(c:Concept)
RETURN DISTINCT c.name AS 의학_개념, c.id AS 개념_ID
```

---

### 2. 증상/질환 분석

#### 2.1 특정 증상을 가진 환자 찾기
```cypher
// "pain"을 언급한 모든 환자
MATCH (p:Patient)-[:SPOKE]->(u:Utterance)
      -[:MENTIONS_CONCEPT]->(c:Concept)
WHERE toLower(c.name) CONTAINS "pain"
RETURN DISTINCT p.id AS 환자_ID, c.name AS 증상
```

#### 2.2 증상 빈도 분석
```cypher
// 가장 많이 언급된 증상/질환
MATCH (u:Utterance)-[:MENTIONS_CONCEPT]->(c:Concept)
RETURN c.name AS 증상, count(*) AS 빈도
ORDER BY 빈도 DESC
LIMIT 20
```

#### 2.3 증상-증상 동시 발생 패턴
```cypher
// 같은 환자가 언급한 증상 쌍
MATCH (p:Patient)-[:SPOKE]->(u1:Utterance)-[:MENTIONS_CONCEPT]->(c1:Concept)
MATCH (p)-[:SPOKE]->(u2:Utterance)-[:MENTIONS_CONCEPT]->(c2:Concept)
WHERE c1 <> c2
RETURN c1.name AS 증상1, c2.name AS 증상2, count(DISTINCT p) AS 환자수
ORDER BY 환자수 DESC
LIMIT 20
```

---

### 3. 감정 분석

#### 3.1 감정 분포
```cypher
// 전체 감정 분포
MATCH (p:Patient)-[:EXPRESSES_PRIMARY]->(e:Emotion)
RETURN e.label AS 감정, count(*) AS 환자수
ORDER BY 환자수 DESC
```

#### 3.2 특정 증상과 연관된 감정
```cypher
// "pain" 증상과 연관된 감정 패턴
MATCH (p:Patient)-[:SPOKE]->(u:Utterance)
      -[:MENTIONS_CONCEPT]->(c:Concept)
WHERE toLower(c.name) CONTAINS "pain"
MATCH (p)-[:EXPRESSES_PRIMARY]->(e:Emotion)
RETURN e.label AS 감정, count(DISTINCT p) AS 환자수
ORDER BY 환자수 DESC
```

#### 3.3 감정-증상 상관관계
```cypher
// 특정 감정을 표현한 환자들이 주로 언급하는 증상
MATCH (p:Patient)-[:EXPRESSES_PRIMARY]->(e:Emotion {label: "anxiety"})
MATCH (p)-[:SPOKE]->(u:Utterance)-[:MENTIONS_CONCEPT]->(c:Concept)
RETURN c.name AS 증상, count(*) AS 빈도
ORDER BY 빈도 DESC
LIMIT 10
```

---

### 4. 의학 지식 탐색 (SNOMED CT)

#### 4.1 개념 계층 탐색
```cypher
// 특정 개념의 상위/하위 개념 찾기
MATCH (c:Concept {name: "Pain"})-[:RELATES_TO*1..2]->(related:Concept)
RETURN related.name AS 관련_개념, length(path) AS 거리
```

#### 4.2 관련 개념 네트워크
```cypher
// 특정 개념과 연결된 모든 개념
MATCH (c:Concept {name: "Pain"})-[:RELATES_TO]-(related:Concept)
RETURN related.name AS 관련_개념, related.id AS 개념_ID
```

#### 4.3 개념 간 경로 찾기
```cypher
// 두 개념 간의 최단 경로
MATCH path = shortestPath(
    (c1:Concept {name: "Pain"})-[*]-(c2:Concept {name: "Anxiety"})
)
RETURN path
```

---

### 5. 환자 그룹 분석

#### 5.1 유사 환자 찾기
```cypher
// 공통 의학 개념을 가진 유사 환자
MATCH (p1:Patient {id: "14"})-[:SPOKE]->(u1:Utterance)
      -[:MENTIONS_CONCEPT]->(c:Concept)
MATCH (p2:Patient)-[:SPOKE]->(u2:Utterance)
      -[:MENTIONS_CONCEPT]->(c)
WHERE p1 <> p2
WITH p2, count(DISTINCT c) AS 공통_개념수
ORDER BY 공통_개념수 DESC
LIMIT 10
RETURN p2.id AS 유사_환자, 공통_개념수
```

#### 5.2 증상 기반 환자 클러스터링
```cypher
// 특정 증상을 가진 환자 그룹
MATCH (p:Patient)-[:SPOKE]->(u:Utterance)
      -[:MENTIONS_CONCEPT]->(c:Concept {name: "Dolour"})
RETURN p.id AS 환자_ID
```

#### 5.3 감정 기반 환자 그룹
```cypher
// 특정 감정을 표현한 환자 그룹
MATCH (p:Patient)-[:EXPRESSES_PRIMARY]->(e:Emotion {label: "sadness"})
RETURN p.id AS 환자_ID
```

---

### 6. 임상 섹션 분석

#### 6.1 섹션별 발화 분석
```cypher
// 특정 섹션의 모든 발화
MATCH (p:Patient)-[:HAS_SECTION]->(s:Section {header: "GENHX"})
MATCH (p)-[:SPOKE]->(u:Utterance)
RETURN u.text AS 발화
```

#### 6.2 섹션별 의학 개념
```cypher
// 특정 섹션에서 언급된 의학 개념
MATCH (p:Patient)-[:HAS_SECTION]->(s:Section {header: "CC"})
MATCH (p)-[:SPOKE]->(u:Utterance)-[:MENTIONS_CONCEPT]->(c:Concept)
RETURN DISTINCT c.name AS 의학_개념
```

---

### 7. 통계 및 인사이트

#### 7.1 전체 통계
```cypher
// 노드 및 관계 통계
MATCH (n)
RETURN labels(n)[0] AS 노드타입, count(n) AS 개수
ORDER BY 개수 DESC
```

#### 7.2 연결성 분석
```cypher
// Concept와 연결된 환자 수
MATCH (c:Concept)<-[:MENTIONS_CONCEPT]-(u:Utterance)<-[:SPOKE]-(p:Patient)
WITH c, count(DISTINCT p) AS 환자수
RETURN c.name AS 개념, 환자수
ORDER BY 환자수 DESC
LIMIT 20
```

#### 7.3 매칭 품질 분석
```cypher
// MENTIONS_CONCEPT 관계의 신뢰도 분포
MATCH ()-[r:MENTIONS_CONCEPT]->()
RETURN r.method AS 매칭방법, 
       avg(r.confidence) AS 평균신뢰도,
       count(*) AS 개수
```

---

### 8. 고급 분석

#### 8.1 임상 경로 탐색
```cypher
// 증상 A → 증상 B로 이어지는 환자 경로
MATCH path = (p:Patient)-[:SPOKE]->(u1:Utterance)
      -[:MENTIONS_CONCEPT]->(c1:Concept {name: "Pain"})
MATCH (p)-[:SPOKE]->(u2:Utterance)
      -[:MENTIONS_CONCEPT]->(c2:Concept {name: "Anxiety"})
RETURN p.id AS 환자, c1.name AS 증상1, c2.name AS 증상2
```

#### 8.2 시간 순서 분석 (발화 순서 기반)
```cypher
// 환자의 발화 순서와 의학 개념 변화
MATCH (p:Patient {id: "14"})-[:SPOKE]->(u:Utterance)
      -[:MENTIONS_CONCEPT]->(c:Concept)
RETURN u.text AS 발화, c.name AS 개념
ORDER BY u.text  // 실제로는 발화 순서 속성이 필요
```

#### 8.3 복합 패턴 발견
```cypher
// 특정 증상 + 특정 감정 조합
MATCH (p:Patient)-[:SPOKE]->(u:Utterance)
      -[:MENTIONS_CONCEPT]->(c:Concept {name: "Pain"})
MATCH (p)-[:EXPRESSES_PRIMARY]->(e:Emotion {label: "anxiety"})
RETURN p.id AS 환자, c.name AS 증상, e.label AS 감정
```

---

## 💡 활용 시나리오

### 시나리오 1: 임상 의사결정 지원
- 환자 증상 입력 → 유사 환자 찾기 → 치료 경과 분석

### 시나리오 2: 증상 패턴 연구
- 특정 증상과 연관된 감정/다른 증상 발견

### 시나리오 3: 환자 그룹핑
- 증상/감정 기반 환자 클러스터링

### 시나리오 4: 의학 지식 확장
- SNOMED CT 관계를 통한 개념 네트워크 탐색

### 시나리오 5: 자연어 질의 응답
- "chest pain을 가진 환자는?" 같은 자연어 질문에 자동 답변

---

## 🔧 실제 사용 예시

Python 코드로 추출:

```python
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()
driver = GraphDatabase.driver(
    os.getenv('NEO4J_URI'),
    auth=(os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))
)

# 예시: 특정 환자의 모든 정보 추출
def extract_patient_info(patient_id: str):
    with driver.session() as session:
        # 기본 정보
        patient = session.run(
            "MATCH (p:Patient {id: $id}) RETURN p",
            id=patient_id
        ).single()
        
        # 발화
        utterances = session.run("""
            MATCH (p:Patient {id: $id})-[:SPOKE]->(u:Utterance)
            RETURN u.text AS text
        """, id=patient_id).data()
        
        # 의학 개념
        concepts = session.run("""
            MATCH (p:Patient {id: $id})-[:SPOKE]->(u:Utterance)
                  -[:MENTIONS_CONCEPT]->(c:Concept)
            RETURN DISTINCT c.name AS concept
        """, id=patient_id).data()
        
        # 감정
        emotions = session.run("""
            MATCH (p:Patient {id: $id})
                  -[:EXPRESSES_PRIMARY|EXPRESSES_SECONDARY|EXPRESSES_ANNOTATED]->(e:Emotion)
            RETURN e.label AS emotion, e.score AS score
        """, id=patient_id).data()
        
        return {
            'patient': patient,
            'utterances': utterances,
            'concepts': concepts,
            'emotions': emotions
        }

# 사용
info = extract_patient_info("14")
print(info)
```

---

## 📈 데이터 규모

- **총 노드**: 약 8,113개
- **총 관계**: 약 11,390개
- **연결된 환자**: 461명 (46.6%)
- **언급된 고유 Concept**: 267개
- **MENTIONS_CONCEPT 관계**: 783개

---

## 🎯 주요 추출 가능 정보 요약

1. ✅ **환자별 정보**: 발화, 감정, 증상, 의학 개념
2. ✅ **증상 분석**: 빈도, 패턴, 동시 발생
3. ✅ **감정 분석**: 분포, 증상-감정 상관관계
4. ✅ **의학 지식**: SNOMED CT 개념 계층, 관련 개념
5. ✅ **환자 그룹**: 유사 환자, 클러스터링
6. ✅ **임상 섹션**: 섹션별 발화 및 개념
7. ✅ **통계**: 전체 통계, 연결성, 매칭 품질
8. ✅ **고급 분석**: 임상 경로, 복합 패턴


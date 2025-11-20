"""
두 KG를 연결하는 스크립트 (LangGraph 버전)

SNOMED CT GPFP KG와 BCB KG를 연결하여
통합 의학 지식 그래프를 구축합니다.
"""

import os
import json
from typing import Dict, List, Any, TypedDict, Optional
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from tqdm import tqdm
from difflib import SequenceMatcher

# 환경변수 로드
load_dotenv()


class ConnectionState(TypedDict):
    """KG 연결 상태 정의"""
    symptom: Dict[str, Any]
    candidate_concepts: List[Dict[str, Any]]
    matched_concept: Optional[Dict[str, Any]]
    match_method: str  # "exact", "similarity", "llm", "none"
    confidence: float
    error: str


class KGLinker:
    """두 KG를 연결하는 클래스"""
    
    def __init__(self):
        # Neo4j 연결 정보
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
        self.neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "kmj15974")
        
        # Neo4j 드라이버
        self.driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_username, self.neo4j_password)
        )
        
        # LLM 설정 (선택적)
        try:
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",  # 비용 효율적인 모델
                temperature=0,
                api_key=os.getenv("OPENAI_API_KEY")
            )
            self.use_llm = True
        except:
            self.use_llm = False
            print("⚠️ LLM을 사용할 수 없습니다. 이름 기반 매칭만 사용합니다.")
    
    def close(self):
        """Neo4j 연결 종료"""
        if self.driver:
            self.driver.close()
    
    def find_candidate_concepts(self, state: ConnectionState) -> ConnectionState:
        """Symptom 이름으로 SNOMED CT Concept 후보 찾기"""
        symptom_name = state["symptom"].get("name", "").lower().strip()
        
        if not symptom_name:
            state["candidate_concepts"] = []
            return state
        
        with self.driver.session() as session:
            # 1. 정확한 이름 매칭
            exact_match = session.run("""
                MATCH (c:Concept)
                WHERE toLower(c.name) = $name
                RETURN c.id AS id, c.name AS name, c.allTerms AS allTerms
                LIMIT 10
            """, name=symptom_name).data()
            
            # 2. allTerms에서 매칭
            if not exact_match:
                exact_match = session.run("""
                    MATCH (c:Concept)
                    WHERE $name IN [term IN c.allTerms WHERE toLower(term) = $name]
                    RETURN c.id AS id, c.name AS name, c.allTerms AS allTerms
                    LIMIT 10
                """, name=symptom_name).data()
            
            # 3. 부분 문자열 매칭 (유사도 기반)
            if not exact_match:
                # 이름에 포함된 개념 찾기
                partial_match = session.run("""
                    MATCH (c:Concept)
                    WHERE toLower(c.name) CONTAINS $name
                       OR any(term IN c.allTerms WHERE toLower(term) CONTAINS $name)
                    RETURN c.id AS id, c.name AS name, c.allTerms AS allTerms
                    LIMIT 20
                """, name=symptom_name).data()
                
                # 유사도 계산하여 정렬
                candidates = []
                for concept in partial_match:
                    concept_name = concept.get("name", "").lower()
                    similarity = SequenceMatcher(None, symptom_name, concept_name).ratio()
                    candidates.append({
                        **concept,
                        "similarity": similarity
                    })
                
                # 유사도 높은 순으로 정렬
                candidates.sort(key=lambda x: x.get("similarity", 0), reverse=True)
                exact_match = candidates[:10]
            
            state["candidate_concepts"] = exact_match
        
        return state
    
    def exact_match(self, state: ConnectionState) -> ConnectionState:
        """정확한 이름 매칭"""
        symptom_name = state["symptom"].get("name", "").lower().strip()
        candidates = state["candidate_concepts"]
        
        if not candidates:
            return state
        
        # 첫 번째 후보가 정확히 일치하면 선택
        first_candidate = candidates[0]
        candidate_name = first_candidate.get("name", "").lower().strip()
        
        if symptom_name == candidate_name:
            state["matched_concept"] = first_candidate
            state["match_method"] = "exact"
            state["confidence"] = 1.0
        elif first_candidate.get("similarity", 0) > 0.95:
            state["matched_concept"] = first_candidate
            state["match_method"] = "similarity"
            state["confidence"] = first_candidate.get("similarity", 0)
        
        return state
    
    def llm_verify_match(self, state: ConnectionState) -> ConnectionState:
        """LLM으로 매칭 검증 (불확실한 경우)"""
        if not self.use_llm or state.get("matched_concept"):
            return state
        
        symptom = state["symptom"]
        candidates = state["candidate_concepts"][:5]  # 상위 5개만 검토
        
        if not candidates:
            return state
        
        # LLM 프롬프트
        prompt = PromptTemplate.from_template("""
다음 증상과 SNOMED CT 의학 개념들을 비교하여 가장 적합한 매칭을 찾아주세요.

증상:
- 이름: {symptom_name}
- 정의: {symptom_definition}
- CUI: {symptom_cui}

SNOMED CT 개념 후보:
{candidates}

각 개념의 이름과 용어를 고려하여, 증상과 가장 의미적으로 일치하는 개념의 ID를 반환해주세요.
정확히 일치하는 것이 없으면 "NONE"을 반환하세요.

응답 형식: JSON
{{
    "matched_id": "개념 ID 또는 NONE",
    "confidence": 0.0-1.0,
    "reason": "매칭 이유"
}}
""")
        
        candidates_text = "\n".join([
            f"- ID: {c['id']}, 이름: {c['name']}, 용어: {', '.join(c.get('allTerms', [])[:3])}"
            for c in candidates
        ])
        
        try:
            response = self.llm.invoke(prompt.format(
                symptom_name=symptom.get("name", ""),
                symptom_definition=symptom.get("definition", ""),
                symptom_cui=symptom.get("cui", ""),
                candidates=candidates_text
            ))
            
            # JSON 파싱
            import re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                matched_id = result.get("matched_id")
                
                if matched_id and matched_id != "NONE":
                    # 매칭된 개념 찾기
                    for candidate in candidates:
                        if candidate["id"] == matched_id:
                            state["matched_concept"] = candidate
                            state["match_method"] = "llm"
                            state["confidence"] = result.get("confidence", 0.7)
                            break
        except Exception as e:
            state["error"] = f"LLM 검증 오류: {e}"
        
        return state
    
    def create_mapping_relationship(self, state: ConnectionState) -> ConnectionState:
        """Symptom과 Concept 간의 매핑 관계 생성"""
        symptom = state["symptom"]
        matched_concept = state.get("matched_concept")
        
        if not matched_concept:
            return state
        
        with self.driver.session() as session:
            # MAPS_TO_SNOMED 관계 생성
            session.run("""
                MATCH (s:Symptom {cui: $cui})
                MATCH (c:Concept {id: $concept_id})
                MERGE (s)-[r:MAPS_TO_SNOMED {
                    method: $method,
                    confidence: $confidence
                }]->(c)
                RETURN r
            """, 
            cui=symptom.get("cui"),
            concept_id=matched_concept["id"],
            method=state.get("match_method", "unknown"),
            confidence=state.get("confidence", 0.0)
            )
        
        return state
    
    def build_langgraph(self) -> StateGraph:
        """LangGraph 워크플로우 구축"""
        workflow = StateGraph(ConnectionState)
        
        # 노드 추가
        workflow.add_node("find_candidates", self.find_candidate_concepts)
        workflow.add_node("exact_match", self.exact_match)
        workflow.add_node("llm_verify", self.llm_verify_match)
        workflow.add_node("create_relationship", self.create_mapping_relationship)
        
        # 엣지 추가
        workflow.set_entry_point("find_candidates")
        workflow.add_edge("find_candidates", "exact_match")
        
        # 정확 매칭이 안 되면 LLM 검증
        def should_use_llm(state: ConnectionState) -> str:
            if state.get("matched_concept"):
                return "create_relationship"
            return "llm_verify"
        
        workflow.add_conditional_edges(
            "exact_match",
            should_use_llm,
            {
                "create_relationship": "create_relationship",
                "llm_verify": "llm_verify"
            }
        )
        
        workflow.add_edge("llm_verify", "create_relationship")
        workflow.add_edge("create_relationship", END)
        
        return workflow.compile()
    
    def link_all_symptoms(self, use_llm_for_uncertain: bool = False):
        """모든 Symptom을 SNOMED CT Concept와 연결"""
        print("="*60)
        print("KG 연결 시작: Symptom → SNOMED CT Concept")
        print("="*60)
        
        # LangGraph 워크플로우 구축
        workflow = self.build_langgraph()
        
        # 모든 Symptom 가져오기
        with self.driver.session() as session:
            symptoms = session.run("""
                MATCH (s:Symptom)
                WHERE NOT (s)-[:MAPS_TO_SNOMED]->()
                RETURN s.cui AS cui, s.name AS name, s.definition AS definition
            """).data()
        
        print(f"\n✓ {len(symptoms)}개의 Symptom 발견 (매핑되지 않은 것)")
        
        # 통계
        stats = {
            "exact": 0,
            "similarity": 0,
            "llm": 0,
            "none": 0
        }
        
        # 각 Symptom에 대해 매칭 수행
        for symptom in tqdm(symptoms, desc="Symptom 매칭"):
            initial_state = {
                "symptom": symptom,
                "candidate_concepts": [],
                "matched_concept": None,
                "match_method": "none",
                "confidence": 0.0,
                "error": ""
            }
            
            try:
                final_state = workflow.invoke(initial_state)
                method = final_state.get("match_method", "none")
                stats[method] = stats.get(method, 0) + 1
            except Exception as e:
                stats["none"] += 1
                print(f"\n[❌ 오류] CUI={symptom.get('cui')} → {e}")
        
        # 결과 출력
        print("\n" + "="*60)
        print("매칭 결과 통계")
        print("="*60)
        print(f"정확 매칭 (exact): {stats['exact']}개")
        print(f"유사도 매칭 (similarity): {stats['similarity']}개")
        print(f"LLM 매칭 (llm): {stats['llm']}개")
        print(f"매칭 실패 (none): {stats['none']}개")
        print(f"총 매핑률: {(stats['exact'] + stats['similarity'] + stats['llm']) / len(symptoms) * 100:.1f}%")
        
        # 연결 통계
        with self.driver.session() as session:
            mapping_count = session.run("""
                MATCH ()-[r:MAPS_TO_SNOMED]->()
                RETURN count(r) AS count
            """).single()["count"]
            
            print(f"\n✓ 총 {mapping_count}개의 MAPS_TO_SNOMED 관계 생성됨")
    
    def get_connection_statistics(self):
        """연결 통계 조회"""
        print("\n=== KG 연결 통계 ===")
        
        with self.driver.session() as session:
            # Symptom → Concept 매핑 수
            mapping_count = session.run("""
                MATCH ()-[r:MAPS_TO_SNOMED]->()
                RETURN count(r) AS count
            """).single()["count"]
            
            # 매핑 방법별 통계
            method_stats = session.run("""
                MATCH ()-[r:MAPS_TO_SNOMED]->()
                RETURN r.method AS method, count(*) AS count
                ORDER BY count DESC
            """).data()
            
            # 평균 신뢰도
            avg_confidence = session.run("""
                MATCH ()-[r:MAPS_TO_SNOMED]->()
                RETURN avg(r.confidence) AS avg_conf
            """).single()["avg_conf"]
            
            print(f"\n총 매핑 수: {mapping_count}개")
            print(f"\n매칭 방법별 통계:")
            for stat in method_stats:
                print(f"  {stat['method']}: {stat['count']}개")
            
            if avg_confidence:
                print(f"\n평균 신뢰도: {avg_confidence:.2f}")
            
            # 예시 쿼리: 환자 → 증상 → SNOMED CT 개념
            example = session.run("""
                MATCH (p:Patient)-[:HAS_SYMPTOM]->(s:Symptom)
                      -[:MAPS_TO_SNOMED]->(c:Concept)
                RETURN p.id AS patient_id, s.name AS symptom, c.name AS concept
                LIMIT 5
            """).data()
            
            if example:
                print(f"\n연결 예시:")
                for ex in example:
                    print(f"  환자 {ex['patient_id']} → {ex['symptom']} → {ex['concept']}")


def main():
    """메인 함수"""
    linker = KGLinker()
    
    try:
        # 모든 Symptom을 SNOMED CT Concept와 연결
        linker.link_all_symptoms(use_llm_for_uncertain=True)
        
        # 통계 출력
        linker.get_connection_statistics()
        
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        linker.close()


if __name__ == "__main__":
    main()


"""
하이브리드 방식으로 두 KG를 연결하는 스크립트 (LangGraph 버전)

Utterance에서 의학 용어 추출 → SNOMED CT Concept 매칭
"""

import os
import json
import re
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


class UtteranceConnectionState(TypedDict):
    """Utterance-Concept 연결 상태 정의"""
    utterance_id: str
    utterance_text: str
    extracted_terms: List[str]
    candidate_concepts: List[Dict[str, Any]]
    matched_concepts: List[Dict[str, Any]]
    match_method: str  # "exact", "similarity", "llm", "none"
    confidence: float
    error: str


class HybridKGLinker:
    """하이브리드 방식으로 KG를 연결하는 클래스"""
    
    def __init__(self, similarity_threshold: float = 0.85):
        """
        Args:
            similarity_threshold: 유사도 매칭 임계값 (0.0-1.0)
                - 0.90-0.95: 보수적 (높은 정확도, 낮은 재현율)
                - 0.85: 균형 (권장, 정확도와 재현율의 균형)
                - 0.70-0.80: 공격적 (낮은 정확도, 높은 재현율)
        """
        # Neo4j 연결 정보
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
        self.neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "kmj15974")
        
        # 유사도 임계값 설정
        self.similarity_threshold = similarity_threshold
        
        # Neo4j 드라이버
        self.driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_username, self.neo4j_password)
        )
        
        # LLM 설정
        try:
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
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
    
    def extract_medical_terms(self, state: UtteranceConnectionState) -> UtteranceConnectionState:
        """Utterance 텍스트에서 의학 용어 추출 (LLM 사용)"""
        utterance_text = state["utterance_text"]
        
        if not self.use_llm:
            # LLM 없으면 간단한 키워드 추출
            medical_keywords = self._extract_keywords_simple(utterance_text)
            state["extracted_terms"] = medical_keywords
            return state
        
        # LLM으로 의학 용어 추출
        prompt = PromptTemplate.from_template("""
다음 환자 발화에서 의학적 증상, 질환, 신체 부위, 의학 용어를 추출해주세요.

발화: {utterance}

의학 용어만 추출하고, 일반적인 단어는 제외하세요.
응답 형식: JSON 배열
["용어1", "용어2", "용어3"]
""")
        
        try:
            response = self.llm.invoke(prompt.format(utterance=utterance_text))
            
            # JSON 파싱
            import re
            json_match = re.search(r'\[.*?\]', response.content, re.DOTALL)
            if json_match:
                terms = json.loads(json_match.group())
                # 문자열 정리
                terms = [t.strip().lower() for t in terms if t.strip()]
                state["extracted_terms"] = terms
            else:
                # JSON 파싱 실패 시 간단한 키워드 추출
                state["extracted_terms"] = self._extract_keywords_simple(utterance_text)
        except Exception as e:
            state["error"] = f"의학 용어 추출 오류: {e}"
            state["extracted_terms"] = self._extract_keywords_simple(utterance_text)
        
        return state
    
    def _extract_keywords_simple(self, text: str) -> List[str]:
        """간단한 키워드 추출 (LLM 없을 때)"""
        # 일반적인 의학 키워드 패턴
        medical_patterns = [
            r'\b(pain|ache|hurt|sore|discomfort)\b',
            r'\b(fever|temperature|hot|cold)\b',
            r'\b(cough|sneeze|breath|breathing)\b',
            r'\b(headache|stomach|chest|back|neck)\b',
            r'\b(nausea|vomit|dizzy|tired|weak)\b',
        ]
        
        keywords = []
        text_lower = text.lower()
        for pattern in medical_patterns:
            matches = re.findall(pattern, text_lower)
            keywords.extend(matches)
        
        return list(set(keywords))
    
    def find_snomed_concepts(self, state: UtteranceConnectionState) -> UtteranceConnectionState:
        """추출된 용어로 SNOMED CT Concept 찾기"""
        terms = state["extracted_terms"]
        all_candidates = []
        
        if not terms:
            state["candidate_concepts"] = []
            return state
        
        with self.driver.session() as session:
            for term in terms:
                # 1. 정확한 이름 매칭
                exact_matches = session.run("""
                    MATCH (c:Concept)
                    WHERE toLower(c.name) = $term
                    RETURN c.id AS id, c.name AS name, c.allTerms AS allTerms
                    LIMIT 5
                """, term=term).data()
                
                # 2. allTerms에서 매칭
                if not exact_matches:
                    exact_matches = session.run("""
                        MATCH (c:Concept)
                        WHERE any(term IN c.allTerms WHERE toLower(term) = $term)
                        RETURN c.id AS id, c.name AS name, c.allTerms AS allTerms
                        LIMIT 5
                    """, term=term).data()
                
                # 3. 부분 문자열 매칭
                if not exact_matches:
                    partial_matches = session.run("""
                        MATCH (c:Concept)
                        WHERE toLower(c.name) CONTAINS $term
                           OR any(term IN c.allTerms WHERE toLower(term) CONTAINS $term)
                        RETURN c.id AS id, c.name AS name, c.allTerms AS allTerms
                        LIMIT 10
                    """, term=term).data()
                    
                    # 유사도 계산
                    for concept in partial_matches:
                        concept_name = concept.get("name", "").lower()
                        similarity = SequenceMatcher(None, term, concept_name).ratio()
                        concept["similarity"] = similarity
                        concept["matched_term"] = term
                    
                    partial_matches.sort(key=lambda x: x.get("similarity", 0), reverse=True)
                    exact_matches = partial_matches[:5]
                
                all_candidates.extend(exact_matches)
        
        # 중복 제거 (id 기준)
        seen_ids = set()
        unique_candidates = []
        for candidate in all_candidates:
            if candidate["id"] not in seen_ids:
                seen_ids.add(candidate["id"])
                unique_candidates.append(candidate)
        
        state["candidate_concepts"] = unique_candidates[:20]  # 상위 20개만
        return state
    
    def match_concepts(self, state: UtteranceConnectionState) -> UtteranceConnectionState:
        """Concept 매칭 및 필터링"""
        candidates = state["candidate_concepts"]
        utterance_text = state["utterance_text"].lower()
        matched = []
        
        for candidate in candidates:
            concept_name = candidate.get("name", "").lower()
            all_terms = [t.lower() for t in candidate.get("allTerms", [])]
            
            # 정확 매칭
            if concept_name in utterance_text or any(term in utterance_text for term in all_terms[:5]):
                candidate["match_method"] = "exact"
                candidate["confidence"] = 1.0
                matched.append(candidate)
            # 유사도 매칭
            elif candidate.get("similarity", 0) >= self.similarity_threshold:
                candidate["match_method"] = "similarity"
                candidate["confidence"] = candidate.get("similarity", self.similarity_threshold)
                matched.append(candidate)
        
        # LLM 검증 (불확실한 경우만)
        if not matched and self.use_llm and candidates:
            matched = self._llm_verify_concepts(state, candidates[:5])
        
        state["matched_concepts"] = matched
        return state
    
    def _llm_verify_concepts(self, state: UtteranceConnectionState, candidates: List[Dict]) -> List[Dict]:
        """LLM으로 Concept 매칭 검증"""
        utterance_text = state["utterance_text"]
        
        prompt = PromptTemplate.from_template("""
다음 환자 발화와 SNOMED CT 의학 개념들을 비교하여, 발화에서 언급된 개념을 찾아주세요.

발화: {utterance}

SNOMED CT 개념 후보:
{candidates}

발화에서 실제로 언급된 개념의 ID만 반환하세요. 없으면 빈 배열을 반환하세요.

응답 형식: JSON
{{"matched_ids": ["id1", "id2"]}}
""")
        
        candidates_text = "\n".join([
            f"- ID: {c['id']}, 이름: {c['name']}"
            for c in candidates
        ])
        
        try:
            response = self.llm.invoke(prompt.format(
                utterance=utterance_text,
                candidates=candidates_text
            ))
            
            json_match = re.search(r'\{.*?\}', response.content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                matched_ids = result.get("matched_ids", [])
                
                matched = []
                for candidate in candidates:
                    if candidate["id"] in matched_ids:
                        candidate["match_method"] = "llm"
                        candidate["confidence"] = 0.8
                        matched.append(candidate)
                
                return matched
        except Exception as e:
            state["error"] = f"LLM 검증 오류: {e}"
        
        return []
    
    def create_utterance_concept_relationships(self, state: UtteranceConnectionState) -> UtteranceConnectionState:
        """Utterance와 Concept 간의 관계 생성"""
        utterance_id = state.get("utterance_id")
        matched_concepts = state["matched_concepts"]
        
        if not utterance_id or not matched_concepts:
            return state
        
        with self.driver.session() as session:
            for concept in matched_concepts:
                # MENTIONS_CONCEPT 관계 생성
                session.run("""
                    MATCH (u:Utterance)
                    WHERE id(u) = $utterance_id
                    MATCH (c:Concept {id: $concept_id})
                    MERGE (u)-[r:MENTIONS_CONCEPT {
                        method: $method,
                        confidence: $confidence
                    }]->(c)
                    RETURN r
                """,
                utterance_id=int(utterance_id),
                concept_id=concept["id"],
                method=concept.get("match_method", "unknown"),
                confidence=concept.get("confidence", 0.0)
                )
        
        return state
    
    def build_langgraph(self) -> StateGraph:
        """LangGraph 워크플로우 구축"""
        workflow = StateGraph(UtteranceConnectionState)
        
        # 노드 추가
        workflow.add_node("extract_terms", self.extract_medical_terms)
        workflow.add_node("find_concepts", self.find_snomed_concepts)
        workflow.add_node("match_concepts", self.match_concepts)
        workflow.add_node("create_relationships", self.create_utterance_concept_relationships)
        
        # 엣지 추가 (순차 실행)
        workflow.set_entry_point("extract_terms")
        workflow.add_edge("extract_terms", "find_concepts")
        workflow.add_edge("find_concepts", "match_concepts")
        workflow.add_edge("match_concepts", "create_relationships")
        workflow.add_edge("create_relationships", END)
        
        return workflow.compile()
    
    def link_all_utterances(self):
        """모든 Utterance를 SNOMED CT Concept와 연결"""
        print("="*60)
        print("하이브리드 KG 연결: Utterance → SNOMED CT Concept")
        print("="*60)
        
        # LangGraph 워크플로우 구축
        workflow = self.build_langgraph()
        
        # 모든 Utterance 가져오기 (아직 연결되지 않은 것)
        with self.driver.session() as session:
            utterances = session.run("""
                MATCH (u:Utterance)
                WHERE NOT (u)-[:MENTIONS_CONCEPT]->()
                RETURN id(u) AS id, u.text AS text
            """).data()
        
        print(f"\n✓ {len(utterances)}개의 Utterance 발견 (매핑되지 않은 것)")
        
        # 통계
        stats = {
            "total": len(utterances),
            "matched": 0,
            "no_match": 0,
            "exact": 0,
            "similarity": 0,
            "llm": 0,
            "errors": 0
        }
        
        # 각 Utterance에 대해 매칭 수행
        for utterance in tqdm(utterances, desc="Utterance 매칭"):
            initial_state = {
                "utterance_id": str(utterance["id"]),
                "utterance_text": utterance["text"],
                "extracted_terms": [],
                "candidate_concepts": [],
                "matched_concepts": [],
                "match_method": "none",
                "confidence": 0.0,
                "error": ""
            }
            
            try:
                final_state = workflow.invoke(initial_state)
                matched = final_state.get("matched_concepts", [])
                
                if matched:
                    stats["matched"] += 1
                    for concept in matched:
                        method = concept.get("match_method", "unknown")
                        if method in stats:
                            stats[method] += 1
                else:
                    stats["no_match"] += 1
                    
            except Exception as e:
                stats["errors"] += 1
                print(f"\n[❌ 오류] Utterance ID={utterance['id']} → {e}")
        
        # 결과 출력
        print("\n" + "="*60)
        print("매칭 결과 통계")
        print("="*60)
        print(f"총 Utterance: {stats['total']}개")
        print(f"매칭 성공: {stats['matched']}개 ({stats['matched']/stats['total']*100:.1f}%)")
        print(f"매칭 실패: {stats['no_match']}개")
        print(f"  - 정확 매칭 (exact): {stats['exact']}개")
        print(f"  - 유사도 매칭 (similarity): {stats['similarity']}개")
        print(f"  - LLM 매칭 (llm): {stats['llm']}개")
        print(f"오류: {stats['errors']}개")
        
        # 연결 통계
        with self.driver.session() as session:
            relationship_count = session.run("""
                MATCH ()-[r:MENTIONS_CONCEPT]->()
                RETURN count(r) AS count
            """).single()["count"]
            
            print(f"\n✓ 총 {relationship_count}개의 MENTIONS_CONCEPT 관계 생성됨")
    
    def get_connection_statistics(self):
        """연결 통계 조회"""
        print("\n=== KG 연결 통계 ===")
        
        with self.driver.session() as session:
            # Utterance → Concept 관계 수
            relationship_count = session.run("""
                MATCH ()-[r:MENTIONS_CONCEPT]->()
                RETURN count(r) AS count
            """).single()["count"]
            
            # 매칭 방법별 통계
            method_stats = session.run("""
                MATCH ()-[r:MENTIONS_CONCEPT]->()
                RETURN r.method AS method, count(*) AS count
                ORDER BY count DESC
            """).data()
            
            # 평균 신뢰도
            avg_confidence = session.run("""
                MATCH ()-[r:MENTIONS_CONCEPT]->()
                RETURN avg(r.confidence) AS avg_conf
            """).single()["avg_conf"]
            
            print(f"\n총 관계 수: {relationship_count}개")
            print(f"\n매칭 방법별 통계:")
            for stat in method_stats:
                print(f"  {stat['method']}: {stat['count']}개")
            
            if avg_confidence:
                print(f"\n평균 신뢰도: {avg_confidence:.2f}")
            
            # 예시 쿼리: 환자 → 발화 → SNOMED CT 개념
            example = session.run("""
                MATCH (p:Patient)-[:SPOKE]->(u:Utterance)
                      -[:MENTIONS_CONCEPT]->(c:Concept)
                RETURN p.id AS patient_id, 
                       substring(u.text, 0, 50) AS utterance,
                       c.name AS concept
                LIMIT 5
            """).data()
            
            if example:
                print(f"\n연결 예시:")
                for ex in example:
                    print(f"  환자 {ex['patient_id']}")
                    print(f"    발화: \"{ex['utterance']}...\"")
                    print(f"    → SNOMED CT: {ex['concept']}")


def main():
    """메인 함수"""
    linker = HybridKGLinker()
    
    try:
        # 모든 Utterance를 SNOMED CT Concept와 연결
        linker.link_all_utterances()
        
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


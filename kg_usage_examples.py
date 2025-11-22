"""
통합 KG 활용 예시

구축된 지식 그래프를 활용하는 다양한 방법
"""

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# GraphCypherQAChain은 최신 버전에서 제거되었을 수 있음
# 대신 LLM으로 직접 Cypher 쿼리를 생성하는 방식 사용
GraphCypherQAChain = None

load_dotenv()


class KGUsageExamples:
    """KG 활용 예시 클래스"""
    
    def __init__(self):
        # Neo4j 연결
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"),
            auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", "kmj15974"))
        )
        
        # LangChain Neo4jGraph
        self.graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"),
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "kmj15974")
        )
        
        # LLM
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # GraphCypherQAChain 대신 LLM으로 직접 Cypher 생성
        self.qa_chain = None
    
    def close(self):
        """연결 종료"""
        if self.driver:
            self.driver.close()
    
    # ==================== 1. 자연어 질의 응답 ====================
    
    def natural_language_query(self, question: str):
        """자연어로 질문하면 LLM이 Cypher 생성하고 답변"""
        print(f"\n질문: {question}")
        print("답변:", end=" ")
        
        # LLM으로 Cypher 쿼리 생성
        prompt = PromptTemplate.from_template("""
You are a Cypher query expert. Generate a Cypher query for Neo4j based on the question.

Schema:
- Patient(id)
- Utterance(text)
- Concept(id, name, allTerms)
- Emotion(label)
- Section(header)
Relationships:
- (Patient)-[:SPOKE]->(Utterance)
- (Utterance)-[:MENTIONS_CONCEPT]->(Concept)
- (Patient)-[:EXPRESSES_PRIMARY|EXPRESSES_SECONDARY|EXPRESSES_ANNOTATED]->(Emotion)
- (Patient)-[:HAS_SECTION]->(Section)
- (Concept)-[:RELATES_TO]->(Concept)

Question: {question}

Generate a Cypher query to answer this question. Return only the Cypher query, no explanation.
""")
        
        try:
            # 1. Cypher 쿼리 생성
            response = self.llm.invoke(prompt.format(question=question))
            cypher = response.content.strip()
            
            # ```cypher 또는 ``` 제거
            import re
            cypher = re.sub(r'```(?:cypher)?\s*', '', cypher).strip()
            cypher = cypher.rstrip('```').strip()
            
            # 2. 쿼리 실행
            result = self.graph.query(cypher)
            
            # 3. 결과를 자연어로 변환
            answer_prompt = PromptTemplate.from_template("""
Question: {question}

Query Results: {results}

Based on the query results, provide a clear answer in Korean.
""")
            
            answer_response = self.llm.invoke(answer_prompt.format(
                question=question,
                results=str(result)
            ))
            
            answer = answer_response.content
            print(answer)
            return answer
            
        except Exception as e:
            print(f"오류: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # ==================== 2. 유사 환자 찾기 ====================
    
    def find_similar_patients(self, patient_id: str, top_k: int = 5):
        """특정 환자와 유사한 환자 찾기"""
        print(f"\n환자 {patient_id}와 유사한 환자 찾기 (상위 {top_k}명)")
        
        query = """
        MATCH (p1:Patient {id: $patient_id})-[:SPOKE]->(u1:Utterance)
              -[:MENTIONS_CONCEPT]->(c:Concept)
        MATCH (p2:Patient)-[:SPOKE]->(u2:Utterance)
              -[:MENTIONS_CONCEPT]->(c)
        WHERE p1 <> p2
        WITH p2, count(DISTINCT c) AS common_concepts
        ORDER BY common_concepts DESC
        LIMIT $top_k
        RETURN p2.id AS patient_id, common_concepts AS 공통_의학_개념_수
        """
        
        with self.driver.session() as session:
            results = session.run(query, patient_id=patient_id, top_k=top_k).data()
            
            print("\n유사 환자:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. 환자 {result['patient_id']}: {result['공통_의학_개념_수']}개 공통 개념")
            
            return results
    
    # ==================== 3. 증상-감정 패턴 분석 ====================
    
    def analyze_symptom_emotion_patterns(self, symptom_concept: str):
        """특정 증상과 연관된 감정 패턴 분석"""
        print(f"\n'{symptom_concept}' 증상과 연관된 감정 패턴 분석")
        
        query = """
        MATCH (p:Patient)-[:SPOKE]->(u:Utterance)
              -[:MENTIONS_CONCEPT]->(c:Concept)
        MATCH (p)-[:EXPRESSES_PRIMARY]->(e:Emotion)
        WHERE toLower(c.name) CONTAINS toLower($symptom)
        RETURN e.label AS emotion, count(*) AS frequency
        ORDER BY frequency DESC
        LIMIT 10
        """
        
        with self.driver.session() as session:
            results = session.run(query, symptom=symptom_concept).data()
            
            print("\n감정 분포:")
            for result in results:
                print(f"  {result['emotion']}: {result['frequency']}명")
            
            return results
    
    # ==================== 4. 의학 개념 확장 탐색 ====================
    
    def explore_related_concepts(self, concept_name: str, depth: int = 2):
        """특정 의학 개념의 관련 개념 탐색 (SNOMED CT 관계 활용)"""
        print(f"\n'{concept_name}' 관련 개념 탐색 (깊이: {depth})")
        
        # Neo4j에서는 변수 길이 패턴에 파라미터를 직접 사용할 수 없으므로
        # 쿼리를 동적으로 생성
        if depth > 5:
            depth = 5  # 최대 깊이 제한
        
        query = f"""
        MATCH (start:Concept)
        WHERE toLower(start.name) CONTAINS toLower($concept_name)
        MATCH path = (start)-[:RELATES_TO*1..{depth}]->(related:Concept)
        WHERE related.name IS NOT NULL
        RETURN DISTINCT related.name AS concept, 
               length(path) AS distance,
               start.name AS start_concept
        ORDER BY distance, related.name
        LIMIT 20
        """
        
        with self.driver.session() as session:
            results = session.run(query, concept_name=concept_name).data()
            
            print(f"\n'{concept_name}' 관련 개념:")
            for result in results:
                print(f"  [{result['distance']}단계] {result['concept']}")
            
            return results
    
    # ==================== 5. 환자 그룹 분석 ====================
    
    def analyze_patient_groups_by_symptoms(self, min_patients: int = 5):
        """증상 기반 환자 그룹 분석"""
        print(f"\n증상 기반 환자 그룹 분석 (최소 {min_patients}명)")
        
        query = """
        MATCH (p:Patient)-[:SPOKE]->(u:Utterance)
              -[:MENTIONS_CONCEPT]->(c:Concept)
        WHERE c.name IS NOT NULL
        WITH c.name AS symptom, count(DISTINCT p) AS patient_count
        WHERE patient_count >= $min_patients
        RETURN symptom, patient_count
        ORDER BY patient_count DESC
        LIMIT 20
        """
        
        with self.driver.session() as session:
            results = session.run(query, min_patients=min_patients).data()
            
            print("\n주요 증상별 환자 수:")
            for result in results:
                print(f"  {result['symptom']}: {result['patient_count']}명")
            
            return results
    
    # ==================== 6. 임상 경로 탐색 ====================
    
    def find_clinical_pathways(self, start_symptom: str, target_symptom: str):
        """증상 A에서 증상 B로 이어지는 임상 경로 찾기"""
        print(f"\n'{start_symptom}' → '{target_symptom}' 임상 경로 탐색")
        
        query = """
        MATCH (c1:Concept)
        WHERE toLower(c1.name) CONTAINS toLower($start)
        MATCH (c2:Concept)
        WHERE toLower(c2.name) CONTAINS toLower($target)
        MATCH path = shortestPath((c1)-[:RELATES_TO*..5]-(c2))
        RETURN [node in nodes(path) | node.name] AS pathway,
               length(path) AS steps
        LIMIT 5
        """
        
        with self.driver.session() as session:
            results = session.run(query, start=start_symptom, target=target_symptom).data()
            
            print("\n임상 경로:")
            for i, result in enumerate(results, 1):
                pathway = " → ".join([p for p in result['pathway'] if p])
                print(f"  {i}. [{result['steps']}단계] {pathway}")
            
            return results
    
    # ==================== 7. 환자 발화에서 의학 개념 추출 ====================
    
    def extract_concepts_from_patient(self, patient_id: str):
        """특정 환자의 발화에서 언급된 모든 의학 개념 추출"""
        print(f"\n환자 {patient_id}의 발화에서 언급된 의학 개념")
        
        query = """
        MATCH (p:Patient {id: $patient_id})-[:SPOKE]->(u:Utterance)
              -[r:MENTIONS_CONCEPT]->(c:Concept)
        RETURN DISTINCT c.name AS concept, 
               r.method AS match_method,
               r.confidence AS confidence
        ORDER BY r.confidence DESC, c.name
        """
        
        with self.driver.session() as session:
            results = session.run(query, patient_id=patient_id).data()
            
            print(f"\n총 {len(results)}개의 의학 개념:")
            for result in results:
                print(f"  - {result['concept']} ({result['match_method']}, 신뢰도: {result['confidence']:.2f})")
            
            return results
    
    # ==================== 8. 통계 대시보드 ====================
    
    def show_kg_statistics(self):
        """KG 전체 통계"""
        print("\n" + "="*60)
        print("통합 KG 통계")
        print("="*60)
        
        with self.driver.session() as session:
            # 노드 통계
            node_stats = session.run("""
                MATCH (n)
                RETURN labels(n)[0] AS label, count(n) AS count
                ORDER BY count DESC
            """).data()
            
            # 관계 통계
            rel_stats = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) AS type, count(r) AS count
                ORDER BY count DESC
            """).data()
            
            # 연결 통계
            connection_stats = session.run("""
                MATCH (p:Patient)-[:SPOKE]->(u:Utterance)
                      -[:MENTIONS_CONCEPT]->(c:Concept)
                RETURN count(DISTINCT p) AS patients_with_concepts,
                       count(DISTINCT u) AS utterances_with_concepts,
                       count(DISTINCT c) AS unique_concepts
            """).single()
            
            print("\n노드 통계:")
            for stat in node_stats:
                print(f"  {stat['label']}: {stat['count']}개")
            
            print("\n관계 통계:")
            for stat in rel_stats:
                print(f"  {stat['type']}: {stat['count']}개")
            
            print("\n연결 통계:")
            print(f"  Concept와 연결된 환자: {connection_stats['patients_with_concepts']}명")
            print(f"  Concept와 연결된 발화: {connection_stats['utterances_with_concepts']}개")
            print(f"  언급된 고유 Concept: {connection_stats['unique_concepts']}개")


def main():
    """사용 예시"""
    kg = KGUsageExamples()
    
    try:
        # 1. 전체 통계
        kg.show_kg_statistics()
        
        # 2. 자연어 질의 응답
        print("\n" + "="*60)
        print("자연어 질의 응답 예시")
        print("="*60)
        kg.natural_language_query("chest pain을 가진 환자는 몇 명인가요?")
        kg.natural_language_query("anxiety를 표현한 환자들이 주로 언급하는 증상은?")
        
        # 3. 유사 환자 찾기
        print("\n" + "="*60)
        print("유사 환자 찾기 예시")
        print("="*60)
        kg.find_similar_patients("14", top_k=5)
        
        # 4. 증상-감정 패턴
        print("\n" + "="*60)
        print("증상-감정 패턴 분석 예시")
        print("="*60)
        kg.analyze_symptom_emotion_patterns("pain")
        
        # 5. 의학 개념 확장
        print("\n" + "="*60)
        print("의학 개념 확장 탐색 예시")
        print("="*60)
        kg.explore_related_concepts("pain", depth=2)
        
        # 6. 환자 그룹 분석
        print("\n" + "="*60)
        print("환자 그룹 분석 예시")
        print("="*60)
        kg.analyze_patient_groups_by_symptoms(min_patients=3)
        
        # 7. 환자별 개념 추출
        print("\n" + "="*60)
        print("환자별 의학 개념 추출 예시")
        print("="*60)
        kg.extract_concepts_from_patient("14")
        
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        kg.close()


if __name__ == "__main__":
    main()


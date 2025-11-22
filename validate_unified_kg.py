"""
통합 KG 검증 스크립트

구축된 통합 KG의 데이터 무결성과 개수를 확인합니다.
"""

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from typing import Dict, List

load_dotenv()


class UnifiedKGValidator:
    """통합 KG 검증 클래스"""
    
    def __init__(self):
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"),
            auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", "kmj15974"))
        )
    
    def close(self):
        """연결 종료"""
        if self.driver:
            self.driver.close()
    
    def validate_node_counts(self):
        """노드 개수 검증"""
        print("\n" + "="*60)
        print("노드 개수 검증")
        print("="*60)
        
        with self.driver.session() as session:
            # 전체 노드 개수
            total_nodes = session.run("MATCH (n) RETURN count(n) AS count").single()["count"]
            print(f"\n전체 노드: {total_nodes}개")
            
            # 노드 타입별 개수
            node_counts = session.run("""
                MATCH (n)
                RETURN labels(n)[0] AS label, count(n) AS count
                ORDER BY count DESC
            """).data()
            
            print("\n노드 타입별 개수:")
            for record in node_counts:
                print(f"  {record['label']}: {record['count']}개")
            
            # 예상 개수와 비교
            expected = {
                "Concept": 4253,  # SNOMED CT GPFP
                "Patient": 989,  # BCB KG
                "Utterance": 989,  # BCB KG
                "Emotion": None,  # 동적
                "SubtleEmotion": None,  # 동적
                "Explanation": None,  # 동적
                "Section": 20  # 예상 (GENHX, CC 등)
            }
            
            print("\n예상 vs 실제 비교:")
            actual_counts = {r['label']: r['count'] for r in node_counts}
            for label, expected_count in expected.items():
                actual = actual_counts.get(label, 0)
                if expected_count is not None:
                    status = "✓" if actual == expected_count else "⚠"
                    print(f"  {status} {label}: 예상 {expected_count}개, 실제 {actual}개")
                else:
                    print(f"  - {label}: {actual}개 (동적)")
            
            return node_counts
    
    def validate_relationship_counts(self):
        """관계 개수 검증"""
        print("\n" + "="*60)
        print("관계 개수 검증")
        print("="*60)
        
        with self.driver.session() as session:
            # 전체 관계 개수
            total_rels = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]
            print(f"\n전체 관계: {total_rels}개")
            
            # 관계 타입별 개수
            rel_counts = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) AS type, count(r) AS count
                ORDER BY count DESC
            """).data()
            
            print("\n관계 타입별 개수:")
            for record in rel_counts:
                print(f"  {record['type']}: {record['count']}개")
            
            return rel_counts
    
    def validate_data_integrity(self):
        """데이터 무결성 검증"""
        print("\n" + "="*60)
        print("데이터 무결성 검증")
        print("="*60)
        
        issues = []
        
        with self.driver.session() as session:
            # 1. Patient-SPOKE-Utterance 관계 검증
            patient_utterance_check = session.run("""
                MATCH (p:Patient)
                OPTIONAL MATCH (p)-[:SPOKE]->(u:Utterance)
                WITH p, count(u) AS utterance_count
                WHERE utterance_count = 0
                RETURN count(p) AS orphaned_patients
            """).single()["orphaned_patients"]
            
            if patient_utterance_check > 0:
                issues.append(f"⚠ 발화가 없는 환자: {patient_utterance_check}명")
            else:
                print("✓ 모든 환자가 발화를 가지고 있습니다")
            
            # 2. Utterance-MENTIONS_CONCEPT 관계 검증
            utterance_concept_check = session.run("""
                MATCH (u:Utterance)
                WHERE NOT (u)-[:MENTIONS_CONCEPT]->()
                RETURN count(u) AS unlinked_utterances
            """).single()["unlinked_utterances"]
            
            if utterance_concept_check > 0:
                issues.append(f"⚠ Concept와 연결되지 않은 발화: {utterance_concept_check}개")
            else:
                print("✓ 모든 발화가 Concept와 연결되어 있습니다")
            
            # 3. Concept-RELATES_TO 관계 검증
            isolated_concepts = session.run("""
                MATCH (c:Concept)
                WHERE NOT (c)-[:RELATES_TO]-()
                RETURN count(c) AS isolated
            """).single()["isolated"]
            
            if isolated_concepts > 0:
                issues.append(f"⚠ 관계가 없는 Concept: {isolated_concepts}개")
            else:
                print("✓ 모든 Concept가 관계를 가지고 있습니다")
            
            # 4. Patient-Emotion 관계 검증
            patient_emotion_check = session.run("""
                MATCH (p:Patient)
                OPTIONAL MATCH (p)-[:EXPRESSES_PRIMARY|EXPRESSES_SECONDARY|EXPRESSES_ANNOTATED]->(e:Emotion)
                WITH p, count(e) AS emotion_count
                WHERE emotion_count = 0
                RETURN count(p) AS no_emotion_patients
            """).single()["no_emotion_patients"]
            
            if patient_emotion_check > 0:
                issues.append(f"⚠ 감정이 없는 환자: {no_emotion_check}명")
            else:
                print("✓ 모든 환자가 감정을 가지고 있습니다")
            
            # 5. Concept name 속성 검증
            concepts_without_name = session.run("""
                MATCH (c:Concept)
                WHERE c.name IS NULL OR c.name = ''
                RETURN count(c) AS no_name
            """).single()["no_name"]
            
            if concepts_without_name > 0:
                issues.append(f"⚠ 이름이 없는 Concept: {concepts_without_name}개")
            else:
                print("✓ 모든 Concept가 이름을 가지고 있습니다")
            
            # 6. 중복 관계 검증
            duplicate_rels = session.run("""
                MATCH (a)-[r:SPOKE]->(b)
                WITH a, b, count(r) AS rel_count
                WHERE rel_count > 1
                RETURN count(*) AS duplicates
            """).single()["duplicates"]
            
            if duplicate_rels > 0:
                issues.append(f"⚠ 중복 관계 발견: {duplicate_rels}개")
            else:
                print("✓ 중복 관계가 없습니다")
        
        if issues:
            print("\n⚠ 발견된 문제:")
            for issue in issues:
                print(f"  {issue}")
        else:
            print("\n✓ 데이터 무결성 검증 통과!")
        
        return issues
    
    def validate_kg_connections(self):
        """KG 간 연결 검증"""
        print("\n" + "="*60)
        print("KG 간 연결 검증")
        print("="*60)
        
        with self.driver.session() as session:
            # Utterance → Concept 연결 통계
            connection_stats = session.run("""
                MATCH (p:Patient)-[:SPOKE]->(u:Utterance)
                      -[:MENTIONS_CONCEPT]->(c:Concept)
                RETURN count(DISTINCT p) AS patients_with_concepts,
                       count(DISTINCT u) AS utterances_with_concepts,
                       count(DISTINCT c) AS unique_concepts,
                       count(*) AS total_connections
            """).single()
            
            print(f"\nBCB KG ↔ SNOMED CT KG 연결:")
            print(f"  Concept와 연결된 환자: {connection_stats['patients_with_concepts']}명")
            print(f"  Concept와 연결된 발화: {connection_stats['utterances_with_concepts']}개")
            print(f"  언급된 고유 Concept: {connection_stats['unique_concepts']}개")
            print(f"  총 연결 관계: {connection_stats['total_connections']}개")
            
            # 연결 비율
            total_patients = session.run("MATCH (p:Patient) RETURN count(p) AS count").single()["count"]
            total_utterances = session.run("MATCH (u:Utterance) RETURN count(u) AS count").single()["count"]
            
            patient_ratio = (connection_stats['patients_with_concepts'] / total_patients * 100) if total_patients > 0 else 0
            utterance_ratio = (connection_stats['utterances_with_concepts'] / total_utterances * 100) if total_utterances > 0 else 0
            
            print(f"\n연결 비율:")
            print(f"  환자 연결률: {patient_ratio:.1f}% ({connection_stats['patients_with_concepts']}/{total_patients})")
            print(f"  발화 연결률: {utterance_ratio:.1f}% ({connection_stats['utterances_with_concepts']}/{total_utterances})")
            
            return connection_stats
    
    def validate_snomed_structure(self):
        """SNOMED CT 구조 검증"""
        print("\n" + "="*60)
        print("SNOMED CT 구조 검증")
        print("="*60)
        
        with self.driver.session() as session:
            # Concept 개수
            concept_count = session.run("MATCH (c:Concept) RETURN count(c) AS count").single()["count"]
            
            # Description 속성 확인
            concepts_with_name = session.run("""
                MATCH (c:Concept)
                WHERE c.name IS NOT NULL
                RETURN count(c) AS count
            """).single()["count"]
            
            # RELATES_TO 관계 개수
            relates_to_count = session.run("""
                MATCH ()-[r:RELATES_TO]->()
                RETURN count(r) AS count
            """).single()["count"]
            
            # IS_A 관계 개수 (typeId = 116680003)
            is_a_count = session.run("""
                MATCH ()-[r:RELATES_TO {typeId: '116680003'}]->()
                RETURN count(r) AS count
            """).single()["count"]
            
            print(f"\nConcept: {concept_count}개")
            print(f"  이름이 있는 Concept: {concepts_with_name}개 ({concepts_with_name/concept_count*100:.1f}%)")
            print(f"RELATES_TO 관계: {relates_to_count}개")
            print(f"  IS_A 관계 (계층): {is_a_count}개 ({is_a_count/relates_to_count*100:.1f}%)")
            
            # 샘플 Concept 확인
            sample_concepts = session.run("""
                MATCH (c:Concept)
                WHERE c.name IS NOT NULL
                RETURN c.id AS id, c.name AS name, size(c.allTerms) AS term_count
                LIMIT 5
            """).data()
            
            print(f"\n샘플 Concept:")
            for concept in sample_concepts:
                print(f"  {concept['id']}: {concept['name']} ({concept['term_count']}개 용어)")
    
    def validate_bcb_structure(self):
        """BCB KG 구조 검증"""
        print("\n" + "="*60)
        print("BCB KG 구조 검증")
        print("="*60)
        
        with self.driver.session() as session:
            # Patient 개수
            patient_count = session.run("MATCH (p:Patient) RETURN count(p) AS count").single()["count"]
            
            # Patient별 평균 발화 수
            avg_utterances = session.run("""
                MATCH (p:Patient)-[:SPOKE]->(u:Utterance)
                WITH p, count(u) AS utterance_count
                RETURN avg(utterance_count) AS avg
            """).single()["avg"]
            
            # Patient별 평균 감정 수
            avg_emotions = session.run("""
                MATCH (p:Patient)-[:EXPRESSES_PRIMARY|EXPRESSES_SECONDARY|EXPRESSES_ANNOTATED]->(e:Emotion)
                WITH p, count(DISTINCT e) AS emotion_count
                RETURN avg(emotion_count) AS avg
            """).single()["avg"]
            
            # Section 개수
            section_count = session.run("MATCH (s:Section) RETURN count(s) AS count").single()["count"]
            
            print(f"\nPatient: {patient_count}명")
            print(f"  평균 발화 수: {avg_utterances:.1f}개")
            print(f"  평균 감정 수: {avg_emotions:.1f}개")
            print(f"Section: {section_count}개")
            
            # Section별 환자 수
            section_patients = session.run("""
                MATCH (s:Section)<-[:HAS_SECTION]-(p:Patient)
                RETURN s.header AS section, count(p) AS patient_count
                ORDER BY patient_count DESC
            """).data()
            
            print(f"\nSection별 환자 수:")
            for section in section_patients:
                print(f"  {section['section']}: {section['patient_count']}명")
    
    def validate_all(self):
        """전체 검증 실행"""
        print("="*60)
        print("통합 KG 검증 시작")
        print("="*60)
        
        # 1. 노드 개수 검증
        node_counts = self.validate_node_counts()
        
        # 2. 관계 개수 검증
        rel_counts = self.validate_relationship_counts()
        
        # 3. 데이터 무결성 검증
        issues = self.validate_data_integrity()
        
        # 4. KG 간 연결 검증
        connection_stats = self.validate_kg_connections()
        
        # 5. SNOMED CT 구조 검증
        self.validate_snomed_structure()
        
        # 6. BCB KG 구조 검증
        self.validate_bcb_structure()
        
        # 최종 요약
        print("\n" + "="*60)
        print("검증 요약")
        print("="*60)
        
        total_nodes = sum(r['count'] for r in node_counts)
        total_rels = sum(r['count'] for r in rel_counts)
        
        print(f"\n전체 노드: {total_nodes}개")
        print(f"전체 관계: {total_rels}개")
        print(f"발견된 문제: {len(issues)}개")
        
        if len(issues) == 0:
            print("\n✓ 모든 검증 통과!")
        else:
            print(f"\n⚠ {len(issues)}개의 문제가 발견되었습니다.")
        
        return {
            "node_counts": node_counts,
            "rel_counts": rel_counts,
            "issues": issues,
            "connection_stats": connection_stats
        }


def main():
    """메인 함수"""
    validator = UnifiedKGValidator()
    
    try:
        validator.validate_all()
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        validator.close()


if __name__ == "__main__":
    main()



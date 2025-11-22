"""
SNOMED CT 메타데이터 속성 사용 분석 스크립트

각 메타데이터 속성이 실제로 필요한지 분석합니다.
"""

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

# Neo4j 연결
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"),
    auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", "kmj15974"))
)

def analyze_metadata_usage():
    """메타데이터 속성 사용 분석"""
    
    with driver.session() as session:
        print("="*60)
        print("SNOMED CT 메타데이터 속성 분석")
        print("="*60)
        
        # 1. Concept 노드 속성 분석
        print("\n[1] Concept 노드 속성 분석")
        print("-" * 60)
        
        # 각 속성의 고유값 개수 확인
        properties = ['effectiveTime', 'definitionStatusId', 'moduleId']
        
        for prop in properties:
            query = f"""
            MATCH (c:Concept)
            WHERE c.{prop} IS NOT NULL
            RETURN count(DISTINCT c.{prop}) AS unique_count,
                   count(c.{prop}) AS total_count
            """
            result = session.run(query).single()
            unique = result['unique_count']
            total = result['total_count']
            diversity = (unique / total * 100) if total > 0 else 0
            
            print(f"\n{prop}:")
            print(f"  - 고유값 개수: {unique}")
            print(f"  - 전체 노드 수: {total}")
            print(f"  - 다양성: {diversity:.2f}%")
            
            if diversity < 5:  # 5% 미만이면 거의 동일한 값
                print(f"  ⚠️  거의 동일한 값 - 제거 고려 가능")
            elif diversity > 50:
                print(f"  ✓ 다양한 값 - 유용할 수 있음")
            else:
                print(f"  ℹ️  중간 정도 다양성")
        
        # 2. definitionStatusId 실제 값 확인
        print("\n[2] definitionStatusId 값 분포")
        print("-" * 60)
        query = """
        MATCH (c:Concept)
        WHERE c.definitionStatusId IS NOT NULL
        RETURN c.definitionStatusId AS status, count(*) AS count
        ORDER BY count DESC
        LIMIT 5
        """
        results = session.run(query)
        for record in results:
            status = record['status']
            count = record['count']
            # SNOMED CT 정의 상태 ID 의미
            if status == "900000000000073002":
                meaning = "완전 정의 (Fully defined)"
            elif status == "900000000000074008":
                meaning = "원시 정의 (Primitive)"
            else:
                meaning = "기타"
            print(f"  {status}: {count}개 ({meaning})")
        
        # 3. moduleId 실제 값 확인
        print("\n[3] moduleId 값 분포")
        print("-" * 60)
        query = """
        MATCH (c:Concept)
        WHERE c.moduleId IS NOT NULL
        RETURN c.moduleId AS module, count(*) AS count
        ORDER BY count DESC
        LIMIT 5
        """
        results = session.run(query)
        for record in results:
            module = record['module']
            count = record['count']
            print(f"  {module}: {count}개")
        
        # 4. Relationship 속성 분석
        print("\n[4] Relationship 속성 분석")
        print("-" * 60)
        
        rel_properties = ['relationshipGroup', 'characteristicTypeId']
        
        for prop in rel_properties:
            query = f"""
            MATCH ()-[r:RELATES_TO]->()
            WHERE r.{prop} IS NOT NULL
            RETURN count(DISTINCT r.{prop}) AS unique_count,
                   count(r.{prop}) AS total_count
            """
            result = session.run(query).single()
            unique = result['unique_count']
            total = result['total_count']
            diversity = (unique / total * 100) if total > 0 else 0
            
            print(f"\n{prop}:")
            print(f"  - 고유값 개수: {unique}")
            print(f"  - 전체 관계 수: {total}")
            print(f"  - 다양성: {diversity:.2f}%")
        
        # 5. 실제 쿼리에서 사용되는 속성 확인
        print("\n[5] 실제 사용 시나리오 분석")
        print("-" * 60)
        
        print("\n✓ 필수 속성 (항상 사용):")
        print("  - id: 노드 식별에 필수")
        print("  - name: 개념 이름 표시에 필수")
        print("  - allTerms: 검색 및 매칭에 유용")
        print("  - typeId (관계): 관계 타입 구분에 필수")
        
        print("\n⚠️  선택적 속성 (사용 빈도 낮음):")
        print("  - effectiveTime: 버전 관리용 (대부분 동일한 날짜)")
        print("  - definitionStatusId: 정의 상태 (대부분 동일한 값)")
        print("  - moduleId: 모듈 구분 (GPFP는 대부분 동일한 모듈)")
        print("  - relationshipGroup: 관계 그룹 (대부분 '0')")
        print("  - characteristicTypeId: 관계 특성 (대부분 동일한 값)")
        
        # 6. 저장 공간 추정
        print("\n[6] 저장 공간 영향 추정")
        print("-" * 60)
        
        # 노드 수와 관계 수 확인
        node_count = session.run("MATCH (c:Concept) RETURN count(c) AS count").single()['count']
        rel_count = session.run("MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS count").single()['count']
        
        # 각 속성의 평균 크기 추정 (바이트)
        estimated_sizes = {
            'effectiveTime': 8,  # "20020131" = 8 bytes
            'definitionStatusId': 18,  # "900000000000073002" = 18 bytes
            'moduleId': 18,  # "900000000000207008" = 18 bytes
            'relationshipGroup': 1,  # "0" = 1 byte
            'characteristicTypeId': 18,  # "900000000000011006" = 18 bytes
        }
        
        total_saved = 0
        for prop, size in estimated_sizes.items():
            if prop in ['effectiveTime', 'definitionStatusId', 'moduleId']:
                saved = node_count * size
            else:
                saved = rel_count * size
            total_saved += saved
            print(f"  {prop}: 약 {saved / 1024 / 1024:.2f} MB")
        
        print(f"\n  총 절약 가능 공간: 약 {total_saved / 1024 / 1024:.2f} MB")
        
        print("\n" + "="*60)
        print("권장사항")
        print("="*60)
        print("""
1. 필수 유지 속성:
   - id, name, allTerms (Concept 노드)
   - id, typeId (RELATES_TO 관계)

2. 제거 고려 속성:
   - effectiveTime: 버전 관리가 필요 없으면 제거
   - definitionStatusId: 대부분 동일한 값
   - moduleId: GPFP는 대부분 동일한 모듈
   - relationshipGroup: 대부분 '0'
   - characteristicTypeId: 대부분 동일한 값

3. 성능 최적화:
   - 불필요한 속성 제거로 인덱스 크기 감소
   - 쿼리 속도 향상
   - 메모리 사용량 감소
        """)

if __name__ == "__main__":
    try:
        analyze_metadata_usage()
    finally:
        driver.close()



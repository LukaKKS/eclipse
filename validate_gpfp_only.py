"""
GPFP에서 실제로 사용하는 개념만 검증하는 스크립트

GPFP Refset 파일에서 referencedComponentId를 추출하여
실제로 사용하는 개념들만 확인합니다.
"""

import os
import csv
from collections import defaultdict, Counter
from typing import Dict, Set, List
from tqdm import tqdm

# CSV 필드 크기 제한 증가
csv.field_size_limit(10000000)


class GPFPValidator:
    """GPFP 전용 검증 클래스"""
    
    def __init__(self, international_path: str, gpfp_path: str):
        self.international_path = international_path
        self.gpfp_path = gpfp_path
        
        # GPFP에서 사용하는 개념 ID들
        self.gpfp_used_concepts: Set[str] = set()
        
        # International에서 로드한 데이터
        self.international_concepts: Dict[str, Dict] = {}
        self.international_descriptions: Dict[str, List[Dict]] = defaultdict(list)
        self.international_relationships: List[Dict] = []
        
        # GPFP 관련 개념들 (GPFP 모듈의 개념 + GPFP Refset의 개념)
        self.gpfp_module_concepts: Set[str] = set()
        
    def extract_gpfp_refset_concepts(self):
        """GPFP Refset에서 사용하는 개념 추출"""
        refset_path = os.path.join(
            self.gpfp_path,
            "Snapshot",
            "Refset",
            "Content",
            "der2_Refset_GPFPSimpleSnapshot_INT_20250101.txt"
        )
        
        print("=== GPFP Refset 개념 추출 ===")
        
        if not os.path.exists(refset_path):
            print(f"⚠ Refset 파일을 찾을 수 없습니다: {refset_path}")
            return
        
        with open(refset_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in tqdm(reader, desc="GPFP Refset 읽는 중"):
                if row['active'] == '1':
                    concept_id = row['referencedComponentId']
                    self.gpfp_used_concepts.add(concept_id)
        
        print(f"✓ GPFP Refset에서 {len(self.gpfp_used_concepts)}개의 개념 발견")
    
    def load_gpfp_module_concepts(self):
        """GPFP 모듈 자체의 개념 로드"""
        concept_path = os.path.join(
            self.gpfp_path,
            "Snapshot",
            "Terminology",
            "sct2_Concept_GPFPSnapshot_INT_20250101.txt"
        )
        
        print("\n=== GPFP 모듈 개념 로드 ===")
        
        if os.path.exists(concept_path):
            with open(concept_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    if row['active'] == '1':
                        self.gpfp_module_concepts.add(row['id'])
                        self.gpfp_used_concepts.add(row['id'])  # GPFP 모듈 개념도 포함
        
        print(f"✓ GPFP 모듈 개념: {len(self.gpfp_module_concepts)}개")
    
    def load_international_data_for_gpfp(self):
        """GPFP에서 사용하는 개념에 대한 International 데이터만 로드"""
        print("\n=== International 데이터 로드 (GPFP 관련만) ===")
        
        # Concept 파일
        concept_path = os.path.join(
            self.international_path,
            "Snapshot",
            "Terminology",
            "sct2_Concept_Snapshot_INT_20251101.txt"
        )
        
        print("Concept 파일 로드 중...")
        with open(concept_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in tqdm(reader, desc="Concepts"):
                concept_id = row['id']
                if concept_id in self.gpfp_used_concepts and row['active'] == '1':
                    self.international_concepts[concept_id] = {
                        'id': concept_id,
                        'effectiveTime': row['effectiveTime'],
                        'moduleId': row['moduleId'],
                        'definitionStatusId': row['definitionStatusId']
                    }
        
        print(f"✓ {len(self.international_concepts)}개의 GPFP 관련 개념 로드됨")
        
        # Description 파일
        desc_path = os.path.join(
            self.international_path,
            "Snapshot",
            "Terminology",
            "sct2_Description_Snapshot-en_INT_20251101.txt"
        )
        
        print("\nDescription 파일 로드 중...")
        with open(desc_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in tqdm(reader, desc="Descriptions"):
                concept_id = row['conceptId']
                if concept_id in self.gpfp_used_concepts and row['active'] == '1':
                    self.international_descriptions[concept_id].append({
                        'id': row['id'],
                        'term': row['term'],
                        'typeId': row['typeId'],
                        'languageCode': row['languageCode']
                    })
        
        total_descriptions = sum(len(descs) for descs in self.international_descriptions.values())
        print(f"✓ {total_descriptions}개의 설명 로드됨")
        
        # Relationship 파일 - GPFP 개념과 관련된 관계만
        rel_path = os.path.join(
            self.international_path,
            "Snapshot",
            "Terminology",
            "sct2_Relationship_Snapshot_INT_20251101.txt"
        )
        
        print("\nRelationship 파일 로드 중 (GPFP 관련만)...")
        print("  (source 또는 destination이 GPFP 개념인 관계만 포함)")
        
        with open(rel_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in tqdm(reader, desc="Relationships"):
                if row['active'] == '1':
                    source_id = row['sourceId']
                    dest_id = row['destinationId']
                    
                    # source 또는 destination 중 하나라도 GPFP 개념이면 포함
                    if (source_id in self.gpfp_used_concepts or 
                        dest_id in self.gpfp_used_concepts):
                        self.international_relationships.append({
                            'id': row['id'],
                            'sourceId': source_id,
                            'destinationId': dest_id,
                            'typeId': row['typeId']
                        })
        
        print(f"✓ {len(self.international_relationships)}개의 관계 로드됨")
    
    def validate_gpfp_data(self):
        """GPFP 데이터 검증"""
        print("\n" + "="*60)
        print("GPFP 데이터 검증 결과")
        print("="*60)
        
        # 1. GPFP Refset 개념이 International에 존재하는지
        missing_concepts = self.gpfp_used_concepts - set(self.international_concepts.keys())
        
        print(f"\n[1] GPFP Refset 개념 검증")
        print(f"  총 GPFP 사용 개념: {len(self.gpfp_used_concepts)}개")
        print(f"  International에 존재: {len(self.international_concepts)}개")
        print(f"  International에 없음: {len(missing_concepts)}개")
        
        if missing_concepts:
            print(f"  ⚠ 다음 개념들이 International에 없습니다:")
            for concept_id in list(missing_concepts)[:10]:
                print(f"    - {concept_id}")
        
        # 2. 설명 검증
        concepts_without_desc = set(self.international_concepts.keys()) - set(self.international_descriptions.keys())
        
        print(f"\n[2] 설명 검증")
        print(f"  설명이 있는 개념: {len(self.international_descriptions)}개")
        print(f"  설명이 없는 개념: {len(concepts_without_desc)}개")
        
        if concepts_without_desc:
            print(f"  ⚠ 설명이 없는 개념들:")
            for concept_id in list(concepts_without_desc)[:10]:
                print(f"    - {concept_id}")
        
        # 3. 관계 검증
        print(f"\n[3] 관계 검증")
        print(f"  총 관계 수: {len(self.international_relationships)}개")
        
        # 관계 타입 분포
        rel_types = Counter(rel['typeId'] for rel in self.international_relationships)
        print(f"\n  관계 타입 분포 (상위 10개):")
        for rel_type, count in rel_types.most_common(10):
            print(f"    {rel_type}: {count}개")
        
        # 4. 통계
        print(f"\n[4] 최종 통계")
        print(f"  GPFP 사용 개념: {len(self.gpfp_used_concepts)}개")
        print(f"  International에서 로드된 개념: {len(self.international_concepts)}개")
        print(f"  설명 수: {sum(len(descs) for descs in self.international_descriptions.values())}개")
        print(f"  관계 수: {len(self.international_relationships)}개")
        
        # 5. KG 구축 가능 여부
        print(f"\n[5] KG 구축 가능 여부")
        if len(self.international_concepts) > 0 and len(self.international_relationships) > 0:
            print(f"  ✓ KG 구축 가능!")
            print(f"  - 노드: {len(self.international_concepts)}개")
            print(f"  - 엣지: {len(self.international_relationships)}개")
        else:
            print(f"  ✗ KG 구축 불가 (데이터 부족)")
        
        print("="*60)
    
    def get_gpfp_concept_ids(self) -> Set[str]:
        """GPFP에서 사용하는 개념 ID 반환"""
        return self.gpfp_used_concepts
    
    def run_full_validation(self):
        """전체 검증 실행"""
        print("="*60)
        print("GPFP 전용 데이터 검증 시작")
        print("="*60)
        
        # 1. GPFP Refset에서 개념 추출
        self.extract_gpfp_refset_concepts()
        
        # 2. GPFP 모듈 개념 로드
        self.load_gpfp_module_concepts()
        
        # 3. International에서 GPFP 관련 데이터만 로드
        self.load_international_data_for_gpfp()
        
        # 4. 검증
        self.validate_gpfp_data()
        
        return {
            'gpfp_concepts': len(self.gpfp_used_concepts),
            'loaded_concepts': len(self.international_concepts),
            'descriptions': sum(len(descs) for descs in self.international_descriptions.values()),
            'relationships': len(self.international_relationships)
        }


def main():
    """메인 함수"""
    international_path = "/Users/giseong/Desktop/ pakdd(jh)/SnomedCT_InternationalRF2_PRODUCTION_20251101T120000Z"
    gpfp_path = "/Users/giseong/Desktop/ pakdd(jh)/SnomedCT_GPFP_PRODUCTION_20250331T120000Z"
    
    validator = GPFPValidator(international_path, gpfp_path)
    results = validator.run_full_validation()
    
    print(f"\n검증 완료!")
    print(f"GPFP에서 사용하는 개념: {results['gpfp_concepts']}개")
    print(f"KG 구축 시 노드 수: {results['loaded_concepts']}개")
    print(f"KG 구축 시 엣지 수: {results['relationships']}개")


if __name__ == "__main__":
    main()


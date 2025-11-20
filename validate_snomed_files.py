"""
SNOMED CT 파일 검증 스크립트

이 스크립트는 다음을 검증합니다:
1. 파일 형식 및 헤더 확인
2. 파일 간 참조 관계 일치 여부 (Concept, Description, Relationship)
3. GPFP와 International 폴더 간의 관계
4. 데이터 무결성 검사
"""

import os
import csv
from collections import defaultdict, Counter
from typing import Dict, Set, List, Tuple
from tqdm import tqdm

# CSV 필드 크기 제한 증가 (SNOMED CT 파일의 큰 필드 처리)
csv.field_size_limit(10000000)  # 10MB로 증가


class SNOMEDFileValidator:
    """SNOMED CT 파일 검증 클래스"""
    
    def __init__(self, international_path: str, gpfp_path: str = None):
        self.international_path = international_path
        self.gpfp_path = gpfp_path
        
        # 데이터 저장
        self.international_concepts: Set[str] = set()
        self.international_descriptions: Dict[str, Set[str]] = defaultdict(set)  # concept_id -> {description_ids}
        self.international_relationships: List[Dict] = []
        
        self.gpfp_concepts: Set[str] = set()
        self.gpfp_descriptions: Dict[str, Set[str]] = defaultdict(set)
        self.gpfp_relationships: List[Dict] = []
        
        # 검증 결과
        self.validation_results = {
            'file_format_checks': {},
            'reference_checks': {},
            'consistency_checks': {},
            'errors': [],
            'warnings': []
        }
    
    def check_file_exists(self, file_path: str, file_name: str) -> bool:
        """파일 존재 여부 확인"""
        exists = os.path.exists(file_path)
        if not exists:
            self.validation_results['errors'].append(f"파일을 찾을 수 없습니다: {file_name}")
        return exists
    
    def check_file_format(self, file_path: str, file_name: str, expected_headers: List[str]) -> Tuple[bool, List[str]]:
        """파일 형식 및 헤더 확인"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                actual_headers = first_line.split('\t')
                
                # 헤더 확인
                missing_headers = set(expected_headers) - set(actual_headers)
                extra_headers = set(actual_headers) - set(expected_headers)
                
                is_valid = len(missing_headers) == 0
                
                if missing_headers:
                    self.validation_results['errors'].append(
                        f"{file_name}: 필수 컬럼이 없습니다: {missing_headers}"
                    )
                if extra_headers:
                    self.validation_results['warnings'].append(
                        f"{file_name}: 추가 컬럼이 있습니다: {extra_headers}"
                    )
                
                return is_valid, actual_headers
        except Exception as e:
            self.validation_results['errors'].append(f"{file_name}: 파일 읽기 오류 - {str(e)}")
            return False, []
    
    def load_international_concepts(self):
        """International Concept 파일 로드"""
        file_path = os.path.join(
            self.international_path, 
            "Snapshot", 
            "Terminology", 
            "sct2_Concept_Snapshot_INT_20251101.txt"
        )
        
        file_name = "International Concept"
        if not self.check_file_exists(file_path, file_name):
            return
        
        expected_headers = ['id', 'effectiveTime', 'active', 'moduleId', 'definitionStatusId']
        is_valid, headers = self.check_file_format(file_path, file_name, expected_headers)
        
        if not is_valid:
            return
        
        print(f"\n{file_name} 파일 로드 중...")
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in tqdm(reader, desc="Loading concepts"):
                concept_id = row['id']
                if row['active'] == '1':
                    self.international_concepts.add(concept_id)
        
        self.validation_results['file_format_checks'][file_name] = {
            'valid': True,
            'total_concepts': len(self.international_concepts),
            'headers': headers
        }
        print(f"  ✓ {len(self.international_concepts)}개의 활성 개념 로드됨")
    
    def load_international_descriptions(self):
        """International Description 파일 로드 및 검증"""
        file_path = os.path.join(
            self.international_path,
            "Snapshot",
            "Terminology",
            "sct2_Description_Snapshot-en_INT_20251101.txt"
        )
        
        file_name = "International Description"
        if not self.check_file_exists(file_path, file_name):
            return
        
        expected_headers = ['id', 'effectiveTime', 'active', 'moduleId', 'conceptId', 
                          'languageCode', 'typeId', 'term', 'caseSignificanceId']
        is_valid, headers = self.check_file_format(file_path, file_name, expected_headers)
        
        if not is_valid:
            return
        
        print(f"\n{file_name} 파일 로드 및 검증 중...")
        orphan_descriptions = []  # conceptId가 Concept에 없는 경우
        inactive_concepts_referenced = set()  # 비활성 개념 참조 확인용
        
        # 전체 Concept 파일에서 비활성 개념도 확인
        concept_file_path = os.path.join(
            self.international_path,
            "Snapshot",
            "Terminology",
            "sct2_Concept_Snapshot_INT_20251101.txt"
        )
        all_concepts = set()  # 활성 + 비활성 모두
        if os.path.exists(concept_file_path):
            with open(concept_file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    all_concepts.add(row['id'])
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in tqdm(reader, desc="Loading descriptions"):
                if row['active'] == '1':
                    concept_id = row['conceptId']
                    description_id = row['id']
                    
                    self.international_descriptions[concept_id].add(description_id)
                    
                    # Concept 존재 여부 확인
                    if concept_id not in self.international_concepts:
                        # 비활성 개념인지 확인
                        if concept_id in all_concepts:
                            inactive_concepts_referenced.add(concept_id)
                        else:
                            orphan_descriptions.append({
                                'description_id': description_id,
                                'concept_id': concept_id,
                                'term': row['term']
                            })
        
        # 검증 결과 저장
        self.validation_results['reference_checks']['description_to_concept'] = {
            'total_descriptions': sum(len(descs) for descs in self.international_descriptions.values()),
            'orphan_descriptions': len(orphan_descriptions),
            'inactive_concepts_referenced': len(inactive_concepts_referenced),
            'orphan_details': orphan_descriptions[:10]  # 처음 10개만
        }
        
        if inactive_concepts_referenced:
            self.validation_results['warnings'].append(
                f"{file_name}: {len(inactive_concepts_referenced)}개의 설명이 비활성 개념을 참조합니다 (정상)"
            )
        
        if orphan_descriptions:
            self.validation_results['warnings'].append(
                f"{file_name}: {len(orphan_descriptions)}개의 설명이 존재하지 않는 개념을 참조합니다"
            )
        
        print(f"  ✓ {sum(len(descs) for descs in self.international_descriptions.values())}개의 활성 설명 로드됨")
        if inactive_concepts_referenced:
            print(f"  ℹ {len(inactive_concepts_referenced)}개의 설명이 비활성 개념을 참조 (정상)")
        if orphan_descriptions:
            print(f"  ⚠ {len(orphan_descriptions)}개의 고아 설명 발견 (존재하지 않는 개념 참조)")
    
    def load_international_relationships(self, sample_size: int = 100000):
        """International Relationship 파일 로드 및 검증"""
        file_path = os.path.join(
            self.international_path,
            "Snapshot",
            "Terminology",
            "sct2_Relationship_Snapshot_INT_20251101.txt"
        )
        
        file_name = "International Relationship"
        if not self.check_file_exists(file_path, file_name):
            return
        
        expected_headers = ['id', 'effectiveTime', 'active', 'moduleId', 'sourceId', 
                          'destinationId', 'relationshipGroup', 'typeId', 
                          'characteristicTypeId', 'modifierId']
        is_valid, headers = self.check_file_format(file_path, file_name, expected_headers)
        
        if not is_valid:
            return
        
        print(f"\n{file_name} 파일 로드 및 검증 중... (샘플: {sample_size}개)")
        invalid_relationships = []
        count = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in tqdm(reader, desc="Loading relationships"):
                if row['active'] == '1':
                    source_id = row['sourceId']
                    dest_id = row['destinationId']
                    
                    # Concept 존재 여부 확인
                    source_exists = source_id in self.international_concepts
                    dest_exists = dest_id in self.international_concepts
                    
                    if source_exists and dest_exists:
                        self.international_relationships.append({
                            'id': row['id'],
                            'sourceId': source_id,
                            'destinationId': dest_id,
                            'typeId': row['typeId']
                        })
                    else:
                        invalid_relationships.append({
                            'id': row['id'],
                            'sourceId': source_id,
                            'destinationId': dest_id,
                            'source_exists': source_exists,
                            'dest_exists': dest_exists
                        })
                    
                    count += 1
                    if count >= sample_size:
                        break
        
        self.validation_results['reference_checks']['relationship_to_concept'] = {
            'total_checked': count,
            'valid_relationships': len(self.international_relationships),
            'invalid_relationships': len(invalid_relationships),
            'invalid_details': invalid_relationships[:10]
        }
        
        if invalid_relationships:
            self.validation_results['warnings'].append(
                f"{file_name}: {len(invalid_relationships)}개의 관계가 존재하지 않는 개념을 참조합니다"
            )
        
        print(f"  ✓ {len(self.international_relationships)}개의 유효한 관계 로드됨")
        if invalid_relationships:
            print(f"  ⚠ {len(invalid_relationships)}개의 무효한 관계 발견")
    
    def load_gpfp_data(self):
        """GPFP 폴더 데이터 로드 및 검증"""
        if not self.gpfp_path:
            return
        
        print("\n=== GPFP 폴더 데이터 검증 ===")
        
        # GPFP Concept
        gpfp_concept_path = os.path.join(
            self.gpfp_path,
            "Snapshot",
            "Terminology",
            "sct2_Concept_GPFPSnapshot_INT_20250101.txt"
        )
        
        if self.check_file_exists(gpfp_concept_path, "GPFP Concept"):
            print("\nGPFP Concept 파일 로드 중...")
            with open(gpfp_concept_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    if row['active'] == '1':
                        self.gpfp_concepts.add(row['id'])
            
            print(f"  ✓ {len(self.gpfp_concepts)}개의 GPFP 개념 로드됨")
        
        # GPFP Description
        gpfp_desc_path = os.path.join(
            self.gpfp_path,
            "Snapshot",
            "Terminology",
            "sct2_Description_GPFPSnapshot-en_INT_20250101.txt"
        )
        
        if self.check_file_exists(gpfp_desc_path, "GPFP Description"):
            print("\nGPFP Description 파일 로드 중...")
            gpfp_orphan_descs = []
            
            with open(gpfp_desc_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    if row['active'] == '1':
                        concept_id = row['conceptId']
                        self.gpfp_descriptions[concept_id].add(row['id'])
                        
                        # International에 존재하는지 확인
                        if concept_id not in self.international_concepts:
                            gpfp_orphan_descs.append(concept_id)
            
            print(f"  ✓ {sum(len(descs) for descs in self.gpfp_descriptions.values())}개의 GPFP 설명 로드됨")
            
            # GPFP와 International 간 관계 확인
            gpfp_in_international = len(self.gpfp_concepts & self.international_concepts)
            gpfp_only = len(self.gpfp_concepts - self.international_concepts)
            
            self.validation_results['consistency_checks']['gpfp_international_overlap'] = {
                'gpfp_concepts': len(self.gpfp_concepts),
                'gpfp_in_international': gpfp_in_international,
                'gpfp_only': gpfp_only,
                'overlap_percentage': (gpfp_in_international / len(self.gpfp_concepts) * 100) if self.gpfp_concepts else 0
            }
            
            if gpfp_only > 0:
                self.validation_results['warnings'].append(
                    f"GPFP: {gpfp_only}개의 개념이 International에 없습니다"
                )
            
            print(f"  ✓ GPFP 개념 중 {gpfp_in_international}개가 International에 존재 ({gpfp_in_international/len(self.gpfp_concepts)*100:.1f}%)")
    
    def check_data_consistency(self):
        """데이터 일관성 검사"""
        print("\n=== 데이터 일관성 검사 ===")
        
        # Concept에 Description이 없는 경우
        concepts_without_desc = self.international_concepts - set(self.international_descriptions.keys())
        
        self.validation_results['consistency_checks']['concepts_without_descriptions'] = {
            'count': len(concepts_without_desc),
            'sample': list(concepts_without_desc)[:10]
        }
        
        if concepts_without_desc:
            self.validation_results['warnings'].append(
                f"{len(concepts_without_desc)}개의 개념에 설명이 없습니다"
            )
            print(f"  ⚠ {len(concepts_without_desc)}개의 개념에 설명이 없음")
        else:
            print(f"  ✓ 모든 개념에 설명이 있음")
        
        # Relationship 타입 분포
        if self.international_relationships:
            rel_types = Counter(rel['typeId'] for rel in self.international_relationships)
            self.validation_results['consistency_checks']['relationship_type_distribution'] = dict(rel_types.most_common(10))
            print(f"  ✓ 관계 타입 분포 확인됨 (상위 10개)")
    
    def generate_report(self):
        """검증 결과 리포트 생성"""
        print("\n" + "="*60)
        print("검증 결과 리포트")
        print("="*60)
        
        # 파일 형식 검사
        print("\n[1] 파일 형식 검사")
        for file_name, result in self.validation_results['file_format_checks'].items():
            status = "✓" if result['valid'] else "✗"
            print(f"  {status} {file_name}: {result.get('total_concepts', 'N/A')}개 항목")
        
        # 참조 관계 검사
        print("\n[2] 참조 관계 검사")
        if 'description_to_concept' in self.validation_results['reference_checks']:
            desc_check = self.validation_results['reference_checks']['description_to_concept']
            status = "✓" if desc_check['orphan_descriptions'] == 0 else "⚠"
            print(f"  {status} Description → Concept: "
                  f"{desc_check['total_descriptions']}개 설명, "
                  f"{desc_check['orphan_descriptions']}개 고아")
        
        if 'relationship_to_concept' in self.validation_results['reference_checks']:
            rel_check = self.validation_results['reference_checks']['relationship_to_concept']
            status = "✓" if rel_check['invalid_relationships'] == 0 else "⚠"
            print(f"  {status} Relationship → Concept: "
                  f"{rel_check['valid_relationships']}개 유효, "
                  f"{rel_check['invalid_relationships']}개 무효")
        
        # 일관성 검사
        print("\n[3] 데이터 일관성 검사")
        if 'concepts_without_descriptions' in self.validation_results['consistency_checks']:
            check = self.validation_results['consistency_checks']['concepts_without_descriptions']
            status = "✓" if check['count'] == 0 else "⚠"
            print(f"  {status} 설명 없는 개념: {check['count']}개")
        
        if 'gpfp_international_overlap' in self.validation_results['consistency_checks']:
            check = self.validation_results['consistency_checks']['gpfp_international_overlap']
            print(f"  ✓ GPFP-International 겹침: {check['gpfp_in_international']}/{check['gpfp_concepts']} "
                  f"({check['overlap_percentage']:.1f}%)")
        
        # 오류 및 경고
        if self.validation_results['errors']:
            print("\n[오류]")
            for error in self.validation_results['errors'][:10]:
                print(f"  ✗ {error}")
        
        if self.validation_results['warnings']:
            print("\n[경고]")
            for warning in self.validation_results['warnings'][:10]:
                print(f"  ⚠ {warning}")
        
        # 최종 요약
        print("\n" + "="*60)
        total_errors = len(self.validation_results['errors'])
        total_warnings = len(self.validation_results['warnings'])
        
        if total_errors == 0 and total_warnings == 0:
            print("✓ 모든 검증 통과!")
        elif total_errors == 0:
            print(f"⚠ {total_warnings}개의 경고가 있습니다 (KG 구축 가능)")
        else:
            print(f"✗ {total_errors}개의 오류가 있습니다 (수정 필요)")
        print("="*60)
    
    def validate_all(self, relationship_sample: int = 100000):
        """전체 검증 실행"""
        print("="*60)
        print("SNOMED CT 파일 검증 시작")
        print("="*60)
        
        # International 데이터 로드
        print("\n=== International 폴더 데이터 검증 ===")
        self.load_international_concepts()
        self.load_international_descriptions()
        self.load_international_relationships(sample_size=relationship_sample)
        
        # GPFP 데이터 로드
        if self.gpfp_path:
            self.load_gpfp_data()
        
        # 일관성 검사
        self.check_data_consistency()
        
        # 리포트 생성
        self.generate_report()
        
        return self.validation_results


def main():
    """메인 함수"""
    international_path = "/Users/giseong/Desktop/ pakdd(jh)/SnomedCT_InternationalRF2_PRODUCTION_20251101T120000Z"
    gpfp_path = "/Users/giseong/Desktop/ pakdd(jh)/SnomedCT_GPFP_PRODUCTION_20250331T120000Z"
    
    validator = SNOMEDFileValidator(international_path, gpfp_path)
    results = validator.validate_all(relationship_sample=100000)
    
    # 결과를 JSON으로 저장
    import json
    output_path = "/Users/giseong/Desktop/ pakdd(jh)/TriageAgent/validation_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n검증 결과가 저장되었습니다: {output_path}")


if __name__ == "__main__":
    main()


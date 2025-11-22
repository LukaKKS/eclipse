"""
Emotion Agent 테스트 스크립트
Test-1 데이터셋에서 3개 샘플을 선택하여 테스트
"""

from emotion_agent import EmotionAgent
import json

def main():
    # Emotion Agent 초기화
    agent = EmotionAgent()
    
    # Test-1 데이터셋에서 3개 샘플 선택
    test_samples = [
        {
            "id": 1,
            "utterance": "An 8-month-old presents with fever, cough, and vomiting. The baby has vomited twice this morning; no diarrhea. Mom states the baby is usually healthy but has \"not been eating well lately.\" Doesn't own a thermometer, but knows the baby is \"hot\" and gave acetaminophen two hours prior to arrival. The baby is wrapped in a blanket, eyes open, appears listless, skin hot and moist, sunken fontanel. Respirations are regular and not labored.",
            "esi_level": 2
        },
        {
            "id": 2,
            "utterance": "17-year-old male, history of suicidality, found unresponsive by parents. There are several bottles of liquor, and unidentified empty pill bottles next to bed",
            "esi_level": 1
        },
        {
            "id": 3,
            "utterance": "15-year-old female presents to ED with her boyfriend claiming, \"I think I'm pregnant. When I told my mom she threw me out of the house and told me never to come back. I have no place to live, and now I might have a baby.\" VS: BP 126/85, HR 100, RR 16, T 98.7°F.",
            "esi_level": 3
        }
    ]
    
    print("=" * 80)
    print("Emotion Agent 테스트 - Test-1 데이터셋 샘플")
    print("=" * 80)
    
    results = []
    
    for sample in test_samples:
        print(f"\n{'='*80}")
        print(f"[샘플 {sample['id']}] ESI Level: {sample['esi_level']}")
        print(f"{'='*80}")
        print(f"\n발화:\n{sample['utterance']}")
        print(f"\n{'─'*80}")
        print("Emotion Agent 분석 중...")
        
        # Emotion Agent 실행
        result = agent.process(sample['utterance'])
        
        # 결과 저장
        results.append({
            "sample_id": sample['id'],
            "esi_level": sample['esi_level'],
            "utterance": sample['utterance'],
            "emotion_result": result
        })
        
        # 결과 출력
        if result["error"]:
            print(f"\n❌ 오류 발생: {result['error']}")
        else:
            print(f"\n✓ Primary Emotion: {result['primary_emotion']}")
            
            if result["secondary_emotion"]:
                print(f"✓ Secondary Emotion: {result['secondary_emotion']} (score: {result['secondary_score']:.2f})")
            else:
                print("✓ Secondary Emotion: None")
            
            if result["subtle_emotion"]:
                print(f"✓ Subtle Emotion: {result['subtle_emotion']} (score: {result['subtle_score']:.2f})")
            else:
                print("✓ Subtle Emotion: None")
            
            if result["masked_emotion"]:
                print(f"✓ Masked Emotion: {result['masked_emotion']}")
            else:
                print("✓ Masked Emotion: None")
            
            print(f"✓ Emotion Risk Score: {result['emotion_risk_score']:.2f}")
            print(f"\nReasoning:\n{result['reasoning']}")
    
    # 전체 결과 요약
    print(f"\n{'='*80}")
    print("전체 결과 요약")
    print(f"{'='*80}\n")
    
    for i, result in enumerate(results, 1):
        if not result["emotion_result"]["error"]:
            print(f"샘플 {i} (ESI {result['esi_level']}):")
            print(f"  - Primary: {result['emotion_result']['primary_emotion']}")
            print(f"  - Risk Score: {result['emotion_result']['emotion_risk_score']:.2f}")
            print()
    
    # JSON 파일로 저장
    output_file = "emotion_agent_test_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 결과가 '{output_file}'에 저장되었습니다.")


if __name__ == "__main__":
    main()


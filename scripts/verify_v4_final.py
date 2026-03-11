
from inference_v4 import EmotionInferenceV4
import pandas as pd

def run_verification():
    print("🧪 Starting Final Verification...")
    engine = EmotionInferenceV4()
    
    test_cases = [
        ("I am so frustrated with this delay!", "Frustrated"),
        ("This food is absolutely disgusting and gross.", "Disgust"),
        ("I'm so excited for the new release!", "Excited"),
        ("The meeting is scheduled at 2 PM.", "Neutral"),
        ("I love spending time with you.", "Love"),
        ("I am very happy today!", "Happy / Joy"),
        ("This is so sad and depressing.", "Sad"),
        ("You make me so angry!", "Angry"),
        ("I am terrified of spiders.", "Fear"),
        ("Wow, that was unexpected!", "Surprise")
    ]
    
    results = []
    passed = 0
    for text, expected in test_cases:
        actual, conf, _ = engine.predict(text)
        status = "✅ PASS" if actual == expected else "❌ FAIL"
        if status == "✅ PASS": passed += 1
        results.append({
            "Text": text,
            "Expected": expected,
            "Actual": actual,
            "Confidence": f"{conf:.1%}",
            "Status": status
        })
        
    df = pd.DataFrame(results)
    print("\nVerification Results:")
    print(df.to_string(index=False))
    
    print(f"\nFinal Score: {passed}/{len(test_cases)}")
    if passed >= 8:
        print("🚀 SUCCESS: High-accuracy 10-class engine is operational!")
    else:
        print("⚠️ WARNING: Some test cases failed. Adjusting rules might be needed.")

if __name__ == "__main__":
    run_verification()

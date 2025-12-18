from sentence_transformers import CrossEncoder

class NewsVerifier:
    def __init__(self):
        print("⏳ Đang load Model NLI (Logic Checking)...")
        # Model này cực tốt cho việc check logic đa ngôn ngữ (bao gồm tiếng Việt)
        # 0: Entailment (Giống/Đúng), 1: Neutral (Trung lập), 2: Contradiction (Sai/Mâu thuẫn)
        self.model = CrossEncoder('symanto/xlm-roberta-base-snli-mnli-anli-xnli')
        print("✅ Model NLI loaded!")

    def check_logic(self, claim, evidence):
        """
        So sánh 1 câu khẳng định (claim) với bằng chứng (evidence)
        Output: Label (ENT/NEU/CON) và Score
        """
        # Input format: (Premise, Hypothesis) -> (Evidence, Claim)
        scores = self.model.predict([(evidence, claim)])
        
        # Lấy nhãn có điểm cao nhất
        label_map = {0: 'TRUE', 1: 'NEUTRAL', 2: 'FAKE'} # Mapping của model symanto
        argmax_idx = scores.argmax()
        
        return label_map[argmax_idx], scores[argmax_idx]

# Test thử tải model
if __name__ == "__main__":
    verifier = NewsVerifier()
    
    # Test case giả định
    evidence = "Bộ Y tế công bố hôm nay có 500 ca mắc mới."
    claim_fake = "Hôm nay có tới 10.000 ca mắc mới theo Bộ Y tế."
    claim_real = "Theo Bộ Y tế, số ca mắc mới trong ngày là 500."
    
    print("\n--- TEST LOGIC ---")
    print(f"Evidence: {evidence}")
    print(f"Claim 1 (Fake): {claim_fake} -> {verifier.check_logic(claim_fake, evidence)}")
    print(f"Claim 2 (Real): {claim_real} -> {verifier.check_logic(claim_real, evidence)}")
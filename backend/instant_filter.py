"""
Instant Fake News Filter - Immediate Detection System
Stops the spread of fake news IMMEDIATELY without waiting for AI model analysis

This module provides instant blocking for known fake news patterns:
- Health miracle claims
- Conspiracy theories
- Known fake news sources
- Dangerous medical misinformation
"""

import re
from typing import Dict, List, Tuple, Optional
from datetime import datetime

class InstantFakeNewsFilter:
    """
    Fast pre-screening system that immediately flags suspicious content
    before expensive AI model processing
    """
    
    def __init__(self):
        self.blocked_patterns = self._load_blocked_patterns()
        self.suspicious_keywords = self._load_suspicious_keywords()
        self.blocked_domains = self._load_blocked_domains()
        
    def _load_blocked_patterns(self) -> List[Dict]:
        """
        Patterns that IMMEDIATELY indicate fake news
        Each pattern has: regex, category, severity, description
        """
        return [
            # === MIRACLE HEALING CLAIMS ===
            {
                "pattern": r"(chữa khỏi|khỏi bệnh|hồi phục|phục hồi).{0,50}(ung thư|cancer|mù|liệt|bại liệt|AIDS|HIV)",
                "category": "MEDICAL_MIRACLE",
                "severity": "CRITICAL",
                "description": "Tuyên bố chữa khỏi bệnh nan y mà không có bằng chứng y khoa"
            },
            {
                "pattern": r"(Pháp Luân|Pháp Luân Công|Falun Gong|Falun Dafa).{0,100}(chữa bệnh|khỏi bệnh|phép màu|kỳ tích|thần tích)",
                "category": "CULT_MEDICAL",
                "severity": "CRITICAL",
                "description": "Tuyên truyền phương pháp chữa bệnh không khoa học"
            },
            {
                "pattern": r"(mù hoàn toàn|bị mù).{0,100}(nhìn thấy|đọc được|trở lại|hồi phục).{0,100}(không dùng thuốc|không phẫu thuật)",
                "category": "MEDICAL_MIRACLE", 
                "severity": "CRITICAL",
                "description": "Tuyên bố hồi phục thị giác mà không có can thiệp y tế"
            },
            
            # === CONSPIRACY THEORIES ===
            {
                "pattern": r"(Đảng Cộng sản|ĐCSTQ).{0,100}(che đậy|âm mưu|bức hại|đàn áp).{0,100}(Pháp Luân|Falun)",
                "category": "POLITICAL_CONSPIRACY",
                "severity": "HIGH",
                "description": "Nội dung tuyên truyền chính trị có xu hướng âm mưu"
            },
            {
                "pattern": r"bác sĩ.{0,50}không thể.{0,50}(chữa|giải thích).{0,50}(phép màu|kỳ tích|thần kỳ)",
                "category": "ANTI_SCIENCE",
                "severity": "HIGH",
                "description": "Phủ nhận y học hiện đại để quảng bá phương pháp không khoa học"
            },
            
            # === DANGEROUS MEDICAL ADVICE ===
            {
                "pattern": r"ngừng (dùng thuốc|điều trị|uống thuốc).{0,100}(khỏe mạnh|bình phục|hết bệnh)",
                "category": "DANGEROUS_MEDICAL",
                "severity": "CRITICAL",
                "description": "Khuyến khích ngừng điều trị y tế - cực kỳ nguy hiểm"
            },
            {
                "pattern": r"không (cần|phải).{0,30}(bác sĩ|bệnh viện|thuốc).{0,50}(chữa khỏi|khỏi bệnh)",
                "category": "DANGEROUS_MEDICAL",
                "severity": "CRITICAL",
                "description": "Ngăn cản việc tìm kiếm chăm sóc y tế chuyên nghiệp"
            },
            
            # === TESTIMONIAL FRAUD ===
            {
                "pattern": r"(70|80|90) tuổi.{0,100}(ung thư|cancer).{0,100}(32|30|25) năm.{0,100}(khỏe mạnh|sống sót)",
                "category": "FAKE_TESTIMONIAL",
                "severity": "HIGH",
                "description": "Câu chuyện cá nhân không thể xác minh về việc sống sót phi thường"
            },
            {
                "pattern": r"(giáo viên|bà|ông|cô).{0,100}(tỉnh|huyện).{0,100}(mù|ung thư).{0,100}(học|luyện).{0,100}(khỏi bệnh|hồi phục)",
                "category": "FAKE_TESTIMONIAL",
                "severity": "HIGH",
                "description": "Lời chứng thực y tế không được kiểm chứng"
            },
            
            # === KNOWN FAKE NEWS SOURCES ===
            {
                "pattern": r"(dkn\.tv|phapluan\.org|epochtimes|epoch times|đại kỷ nguyên)",
                "category": "BLOCKED_SOURCE",
                "severity": "CRITICAL",
                "description": "Nguồn tin được xác định phát tán thông tin sai lệch"
            }
        ]
    
    def _load_suspicious_keywords(self) -> Dict[str, List[str]]:
        """
        Keywords that raise suspicion level (not instant block, but flag for review)
        """
        return {
            "medical_claims": [
                "kỳ tích", "phép màu", "thần tích", "phép lạ", 
                "chữa khỏi hoàn toàn", "bình phục kỳ diệu",
                "bác sĩ không tin", "y học không giải thích được"
            ],
            "cult_related": [
                "Pháp Luân Công", "Falun Gong", "Pháp Luân Đại Pháp",
                "Chân Thiện Nhẫn", "Lý Hồng Chí", "Li Hongzhi"
            ],
            "conspiracy": [
                "âm mưu", "che đậy", "bức hại", "đàn áp",
                "sự thật bị giấu", "họ không muốn bạn biết"
            ],
            "anti_vaccine": [
                "vaccine gây", "tiêm chủng nguy hiểm", "vắc xin độc hại"
            ]
        }
    
    def _load_blocked_domains(self) -> List[str]:
        """
        Domains that are known to spread fake news
        """
        return [
            "dkn.tv",
            "phapluan.org", 
            "epochtimes.fr",
            "epochtimes.com",
            "minghui.org"
        ]
    
    def check(self, text: str, url: Optional[str] = None) -> Dict:
        """
        Main method: Instantly check if content should be blocked
        
        Returns:
            {
                "should_block": bool,
                "severity": "CRITICAL" | "HIGH" | "MEDIUM" | "LOW",
                "reasons": List[str],
                "matched_patterns": List[Dict],
                "suspicion_score": float (0-1)
            }
        """
        text_lower = text.lower()
        
        # Results
        matched_patterns = []
        reasons = []
        max_severity = None
        
        # === STEP 1: Check URL against blocked domains ===
        if url:
            for domain in self.blocked_domains:
                if domain in url.lower():
                    matched_patterns.append({
                        "type": "BLOCKED_DOMAIN",
                        "matched": domain,
                        "severity": "CRITICAL"
                    })
                    reasons.append(f"🚫 Nguồn tin không đáng tin cậy: {domain}")
                    max_severity = "CRITICAL"
        
        # === STEP 2: Check dangerous patterns ===
        for pattern_config in self.blocked_patterns:
            matches = re.finditer(pattern_config["pattern"], text, re.IGNORECASE)
            for match in matches:
                matched_text = match.group(0)
                matched_patterns.append({
                    "type": pattern_config["category"],
                    "matched": matched_text[:100],  # Limit length
                    "severity": pattern_config["severity"],
                    "description": pattern_config["description"]
                })
                reasons.append(f"⚠️ {pattern_config['description']}")
                
                # Update max severity
                if not max_severity or self._severity_level(pattern_config["severity"]) > self._severity_level(max_severity):
                    max_severity = pattern_config["severity"]
        
        # === STEP 3: Calculate suspicion score based on keywords ===
        suspicion_score = self._calculate_suspicion_score(text_lower)
        
        # === DECISION: Should we block? ===
        should_block = (
            max_severity in ["CRITICAL", "HIGH"] or 
            len(matched_patterns) >= 2 or
            suspicion_score >= 0.7
        )
        
        return {
            "should_block": should_block,
            "severity": max_severity or "LOW",
            "reasons": reasons,
            "matched_patterns": matched_patterns,
            "suspicion_score": suspicion_score,
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_suspicion_score(self, text_lower: str) -> float:
        """
        Calculate suspicion score based on keyword frequency
        Returns: 0.0 to 1.0 (higher = more suspicious)
        """
        total_score = 0.0
        max_possible = 0.0
        
        for category, keywords in self.suspicious_keywords.items():
            category_score = sum(1 for kw in keywords if kw.lower() in text_lower)
            
            # Weight different categories
            weight = 1.0
            if category == "medical_claims":
                weight = 1.5
            elif category == "cult_related":
                weight = 2.0
            elif category == "conspiracy":
                weight = 1.2
            elif category == "anti_vaccine":
                weight = 2.0
            
            total_score += category_score * weight
            max_possible += len(keywords) * weight
        
        # Normalize to 0-1
        if max_possible > 0:
            return min(1.0, total_score / (max_possible * 0.3))  # Scale factor
        return 0.0
    
    def _severity_level(self, severity: str) -> int:
        """Convert severity to numeric level for comparison"""
        levels = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
        return levels.get(severity, 0)
    
    def get_warning_message(self, check_result: Dict) -> str:
        """
        Generate user-friendly warning message based on check result
        """
        if not check_result["should_block"]:
            return ""
        
        severity = check_result["severity"]
        
        if severity == "CRITICAL":
            header = "🚨 CẢNH BÁO NGHIÊM TRỌNG - TIN GIẢNGUY HIỂM"
        elif severity == "HIGH":
            header = "⚠️ CẢNH BÁO - THÔNG TIN ĐÁNG NGHI"
        else:
            header = "⚪ LƯU Ý - CẦN KIỂM CHỨNG"
        
        message = f"{header}\n\n"
        message += "Lý do:\n"
        
        for reason in check_result["reasons"][:3]:  # Top 3 reasons
            message += f"• {reason}\n"
        
        message += f"\nĐộ nghi ngờ: {check_result['suspicion_score']*100:.0f}%"
        
        if severity == "CRITICAL":
            message += "\n\n⛔ Nội dung này có thể gây hại đến sức khỏe hoặc an toàn của bạn. Vui lòng tham khảo ý kiến chuyên gia y tế."
        
        return message


# === TEST WITH PROVIDED EXAMPLES ===
if __name__ == "__main__":
    filter_system = InstantFakeNewsFilter()
    
    # Example 1: From phapluan_bien_the.static.html
    example1 = """
    Hành trình chiến thắng ung thư trực tràng giai đoạn cuối của một phụ nữ 70 tuổi
    
    Ryoko Mochizuki là một phụ nữ Nhật Bản gốc Hoa, bà được chẩn đoán mắc bệnh 
    ung thư trực tràng giai đoạn cuối vào năm 1991. Khi ấy các bác sĩ ước tính 
    bà sẽ sống không quá năm năm nữa. Hơn ba thập kỷ sau, ở tuổi 70, bà không 
    chỉ sống sót mà sức khỏe của bà còn cải thiện tốt hơn so với trước khi mắc 
    bệnh ung thư. May mắn thay, bà Mochizuki đã tìm thấy hi vọng trong Pháp Luân Công.
    
    Sau khi bắt đầu tập Pháp Luân Công, sức khỏe của bà Mochizuki đã hồi phục tốt.
    Từ chối nhận trợ cấp quốc gia sau khi đã khỏi bệnh.
    
    Theo Epoch Times
    """
    
    # Example 2: From phapluan_goc.static.html  
    example2 = """
    Cựu giáo viên bị mù, trải nghiệm kỳ tích hồi sinh đôi mắt sáng
    
    Cô Vũ Thị Xuân, giáo viên tiểu học nghỉ hưu, sống tại huyện Bắc Hà, tỉnh Lào Cai. 
    Năm 2020, tôi bị mù cả hai mắt. Bác sĩ nói cứ về khi nào Covid qua đi thì sẽ 
    xuống bệnh viện Bạch Mai để người ta khoét hai mắt của tôi đi để bảo toàn tính mạng.
    
    Khi tôi quyết định tự tử thì tình cờ tôi được một đồng nghiệp giới thiệu cho tôi 
    Pháp Luân Công. Sau một thời gian luyện công, một hôm tôi đã đọc được chữ. 
    Bây giờ tôi đã đọc được sách dù chữ to hay chữ bé tôi cũng đọc được hết.
    
    Từ tháng 4/2020 đến giờ tôi không phải dùng đến bất kể một thứ thuốc gì, 
    tất cả các bệnh trong cơ thể đều biến mất hết.
    
    Quý độc giả có thể tìm hiểu thêm tại website: phapluan.org
    """
    
    print("=" * 70)
    print("INSTANT FAKE NEWS FILTER - TEST WITH 2 EXAMPLES")
    print("=" * 70)
    
    print("\n📄 EXAMPLE 1: Cancer miracle healing (70-year-old woman)")
    print("-" * 70)
    result1 = filter_system.check(example1, url="https://www.dkn.tv/article/123")
    print(f"Should Block: {result1['should_block']}")
    print(f"Severity: {result1['severity']}")
    print(f"Suspicion Score: {result1['suspicion_score']:.2f}")
    print(f"\nMatched Patterns: {len(result1['matched_patterns'])}")
    for pattern in result1['matched_patterns']:
        print(f"  - {pattern['type']}: {pattern.get('description', pattern['matched'])}")
    print(f"\n{filter_system.get_warning_message(result1)}")
    
    print("\n" + "=" * 70)
    print("\n📄 EXAMPLE 2: Blindness cured miracle (Teacher)")
    print("-" * 70)
    result2 = filter_system.check(example2, url="https://phapluan.org/article/456")
    print(f"Should Block: {result2['should_block']}")
    print(f"Severity: {result2['severity']}")
    print(f"Suspicion Score: {result2['suspicion_score']:.2f}")
    print(f"\nMatched Patterns: {len(result2['matched_patterns'])}")
    for pattern in result2['matched_patterns']:
        print(f"  - {pattern['type']}: {pattern.get('description', pattern['matched'])}")
    print(f"\n{filter_system.get_warning_message(result2)}")
    
    print("\n" + "=" * 70)

// Cấu hình API Local
const API_URL = "http://localhost:8000/api/v1";

// 1. HÀM LẤY USER ID (Tạo định danh ẩn danh)
async function getOrCreateUserId() {
    return new Promise((resolve) => {
        chrome.storage.local.get(['user_id'], (result) => {
            if (result.user_id) {
                resolve(result.user_id);
            } else {
                const newId = crypto.randomUUID();
                chrome.storage.local.set({ user_id: newId }, () => {
                    resolve(newId);
                });
            }
        });
    });
}

// 2. LOGIC NÚT KIỂM TRA
document.getElementById('check-btn').addEventListener('click', async () => {
    const statusDiv = document.getElementById('status-msg');
    const resultBox = document.getElementById('main-result');
    const claimsDiv = document.getElementById('claims-list');
    const btn = document.getElementById('check-btn');

    // Reset UI
    statusDiv.textContent = "⏳ Đang đọc báo & gửi về AI...";
    statusDiv.style.display = 'block';
    resultBox.style.display = 'none';
    claimsDiv.innerHTML = '';
    btn.disabled = true;

    // Lấy nội dung trang web
    let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    chrome.scripting.executeScript({
        target: { tabId: tab.id },
        function: getPageContent,
    }, async (results) => {
        if (!results || !results[0] || !results[0].result) {
            statusDiv.textContent = "❌ Không lấy được nội dung bài báo.";
            btn.disabled = false;
            return;
        }

        const pageText = results[0].result;

        try {
            // GỌI API VERIFY
            const response = await fetch(`${API_URL}/verify`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: pageText })
            });

            if (!response.ok) throw new Error('API Error');
            const data = await response.json();

            // HIỂN THỊ KẾT QUẢ TỔNG QUAN
            statusDiv.style.display = 'none';
            resultBox.style.display = 'block';
            resultBox.className = 'status-box ' + data.status.toLowerCase();
            
            document.getElementById('verdict').textContent = data.status === 'FAKE' ? "CẢNH BÁO: TIN GIẢ" : (data.status === 'REAL' ? "TIN CHÍNH XÁC" : "CHƯA XÁC THỰC");
            document.getElementById('confidence').textContent = `Độ tin cậy: ${(data.confidence * 100).toFixed(1)}% | Model: ${data.model_version}`;
            document.getElementById('explanation').textContent = data.explanation;

            // HIỂN THỊ CHI TIẾT TỪNG CÂU + NÚT REPORT
            if (data.details && data.details.length > 0) {
                claimsDiv.innerHTML = "<div style='font-size:11px; margin:5px 0; font-weight:bold;'>Chi tiết kiểm chứng:</div><ul>";
                
                data.details.forEach(item => {
                    const icon = item.status === 'REFUTED' ? '❌' : (item.status === 'SUPPORTED' ? '✅' : '⚪');
                    const claimId = item.claim_id || "null"; // ID để report
                    
                    // Tạo HTML cho từng dòng claim
                    const li = document.createElement('li');
                    li.innerHTML = `
                        <span class="claim-text">${icon} ${item.claim}</span>
                        <div class="actions">
                            <button class="btn-report rep-fake" 
                                onclick="reportClaim('${claimId}', 'FAKE', '${item.status}', ${item.score}, '${data.model_version}')">
                                🚨 Báo sai
                            </button>
                            <button class="btn-report rep-real"
                                onclick="reportClaim('${claimId}', 'REAL', '${item.status}', ${item.score}, '${data.model_version}')">
                                👍 Xác nhận đúng
                            </button>
                        </div>
                    `;
                    claimsDiv.appendChild(li);
                });
                claimsDiv.innerHTML += "</ul>";
            }

        } catch (err) {
            statusDiv.textContent = "❌ Lỗi: " + err.message;
            statusDiv.style.display = 'block';
        } finally {
            btn.disabled = false;
        }
    });
});

// 3. HÀM GỬI BÁO CÁO (REPORT)
// Hàm này phải gắn vào window để gọi được từ onclick trong HTML string
window.reportClaim = async (claimId, feedback, aiLabel, aiConf, modelVer) => {
    if (claimId === "null" || !claimId) {
        alert("⚠️ Câu này chưa có trong Database (Claim ID = null) nên không thể báo cáo.\n\nHãy chạy lại script rebuild_kb.py để nạp dữ liệu chuẩn.");
        return;
    }

    const userId = await getOrCreateUserId();
    const comment = prompt("Bạn có muốn ghi chú gì thêm không? (Không bắt buộc)");

    try {
        const response = await fetch(`${API_URL}/report`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_id: userId,
                claim_id: parseInt(claimId),
                feedback: feedback, // 'FAKE' hoặc 'REAL' (Ý kiến user)
                comment: comment || "",
                ai_label: aiLabel,
                ai_confidence: aiConf,
                model_version: modelVer
            })
        });

        if (response.ok) {
            alert("✅ Cảm ơn! Báo cáo của bạn đã được gửi tới Admin.");
        } else {
            alert("❌ Lỗi gửi báo cáo.");
        }
    } catch (e) {
        alert("❌ Lỗi kết nối: " + e.message);
    }
};

// 4. CONTENT SCRIPT (Chạy trên trang web)
function getPageContent() {
    // Lấy tiêu đề và nội dung bài báo (VnExpress & General)
    const title = document.querySelector('h1.title-detail')?.innerText || document.title;
    const content = document.querySelector('article.fck_detail')?.innerText || document.body.innerText;
    
    // Cắt gọn bớt để gửi cho nhanh
    const fullText = title + ". " + content;
    return fullText.substring(0, 4000); 
}
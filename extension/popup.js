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
            // GỌI API VERIFY (with URL for instant domain checking)
            const response = await fetch(`${API_URL}/verify`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    text: pageText,
                    url: tab.url  // Pass URL for instant blocking
                })
            });

            if (!response.ok) throw new Error('API Error');
            const data = await response.json();

            // HIỂN THỊ KẾT QUẢ TỔNG QUAN
            statusDiv.style.display = 'none';
            resultBox.style.display = 'block';
            resultBox.className = 'status-box ' + data.status.toLowerCase();
            
            // Check if it was instantly blocked
            let verdictText = "";
            if (data.instant_block) {
                verdictText = "🚫 CHẶN NGAY LẬP TỨC - TIN GIẢ NGUY HIỂM";
                resultBox.classList.add('instant-block');
            } else {
                verdictText = data.status === 'FAKE' ? "CẢNH BÁO: TIN GIẢ" : (data.status === 'REAL' ? "TIN CHÍNH XÁC" : "CHƯA XÁC THỰC");
            }
            
            document.getElementById('verdict').textContent = verdictText;
            document.getElementById('confidence').textContent = `Độ tin cậy: ${(data.confidence * 100).toFixed(1)}% | Model: ${data.model_version}`;
            document.getElementById('explanation').textContent = data.explanation;

            // HIỂN THỊ CHI TIẾT TỪNG CÂU + NÚT REPORT
            if (data.details && data.details.length > 0) {
                claimsDiv.innerHTML = "<div style='font-size:11px; margin:5px 0; font-weight:bold;'>Chi tiết kiểm chứng:</div><ul>";
                
                data.details.forEach(item => {
                    const icon = item.status === 'REFUTED' ? '❌' : (item.status === 'SUPPORTED' ? '✅' : '⚪');
                    const claimId = item.claim_id || "null"; 
                    
                    // --- SỬA CHỖ NÀY: Dùng data-attribute thay vì onclick ---
                    const li = document.createElement('li');
                    li.innerHTML = `
                        <span class="claim-text">${icon} ${item.claim}</span>
                        <div class="actions">
                            <button class="btn-report rep-fake" 
                                data-id="${claimId}" 
                                data-feedback="FAKE"
                                data-ailabel="${item.status}"
                                data-aiconf="${item.score}"
                                data-modelver="${data.model_version}">
                                🚨 Báo sai
                            </button>
                            <button class="btn-report rep-real"
                                data-id="${claimId}" 
                                data-feedback="REAL"
                                data-ailabel="${item.status}"
                                data-aiconf="${item.score}"
                                data-modelver="${data.model_version}">
                                👍 Xác nhận đúng
                            </button>
                        </div>
                    `;
                    claimsDiv.appendChild(li);
                });
                claimsDiv.innerHTML += "</ul>";

                // --- GẮN SỰ KIỆN CLICK SAU KHI TẠO HTML ---
                addReportListeners();
            }

        } catch (err) {
            statusDiv.textContent = "❌ Lỗi: " + err.message;
            statusDiv.style.display = 'block';
        } finally {
            btn.disabled = false;
        }
    });
});

// 3. HÀM GẮN SỰ KIỆN CLICK (Thay thế cho onclick)
function addReportListeners() {
    const buttons = document.querySelectorAll('.btn-report');
    buttons.forEach(btn => {
        btn.addEventListener('click', async (e) => {
            // Lấy dữ liệu từ data-attribute
            const target = e.currentTarget; // Nút được bấm
            const claimId = target.dataset.id;
            const feedback = target.dataset.feedback;
            const aiLabel = target.dataset.ailabel;
            const aiConf = target.dataset.aiconf;
            const modelVer = target.dataset.modelver;

            await handleReport(target, claimId, feedback, aiLabel, aiConf, modelVer);
        });
    });
}

// 4. LOGIC GỬI REPORT
async function handleReport(btnElement, claimId, feedback, aiLabel, aiConf, modelVer) {
    if (claimId === "null" || !claimId) {
        alert("⚠️ Câu này chưa có trong Database (Claim ID = null) nên không thể báo cáo.\n\nHãy chạy lại script rebuild_kb.py để nạp dữ liệu chuẩn.");
        return;
    }

    const userId = await getOrCreateUserId();
    
    // Hiệu ứng loading
    const originalText = btnElement.textContent;
    btnElement.textContent = "⏳...";
    btnElement.disabled = true;

    try {
        const response = await fetch(`${API_URL}/report`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_id: userId,
                claim_id: parseInt(claimId),
                feedback: feedback, 
                comment: "Reported via Extension V2",
                ai_label: aiLabel,
                ai_confidence: parseFloat(aiConf),
                model_version: modelVer
            })
        });

        if (response.ok) {
            alert("✅ Đã gửi báo cáo thành công!");
            btnElement.textContent = "Đã gửi";
        } else {
            alert("❌ Lỗi Server.");
            btnElement.textContent = originalText;
            btnElement.disabled = false;
        }
    } catch (e) {
        alert("❌ Lỗi kết nối: " + e.message);
        btnElement.textContent = originalText;
        btnElement.disabled = false;
    }
};

// 5. CONTENT SCRIPT
function getPageContent() {
    const title = document.querySelector('h1.title-detail')?.innerText || document.title;
    const content = document.querySelector('article.fck_detail')?.innerText || document.body.innerText;
    const fullText = title + ". " + content;
    return fullText.substring(0, 4000); 
}
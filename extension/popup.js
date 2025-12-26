document.getElementById('check-btn').addEventListener('click', async () => {
    const statusDiv = document.getElementById('status');
    const resultBox = document.getElementById('result-box');
    const btn = document.getElementById('check-btn');
    const detailsDiv = document.getElementById('details');

    // Reset UI
    statusDiv.textContent = "⏳ Đang đọc nội dung & phân tích...";
    resultBox.style.display = 'none';
    detailsDiv.innerHTML = '';
    btn.disabled = true;

    // 1. Lấy nội dung trang web hiện tại
    let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    chrome.scripting.executeScript({
        target: { tabId: tab.id },
        function: getPageContent,
    }, async (results) => {
        const pageData = results[0].result;
        
        if (!pageData) {
            statusDiv.textContent = "❌ Không lấy được nội dung bài báo.";
            btn.disabled = false;
            return;
        }

        try {
            // 2. Gọi API của bạn (Localhost)
            statusDiv.textContent = "🚀 Đang gửi về AI Server...";
            
            const response = await fetch('http://localhost:8000/api/v1/verify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: `${pageData.title}\n\n${pageData.content}`
                })
            });

            if (!response.ok) throw new Error('API Error');
            const data = await response.json();

            // 3. Hiển thị kết quả
            statusDiv.textContent = "";
            resultBox.style.display = 'block';
            resultBox.className = 'result ' + data.status.toLowerCase();
            
            document.getElementById('verdict').textContent = data.status;
            document.getElementById('confidence').textContent = `Độ tin cậy: ${(data.confidence * 100).toFixed(1)}%`;
            
            detailsDiv.innerHTML = `<b>📝 Giải thích:</b> ${data.explanation}<br><br>`;
            
            // Hiển thị chi tiết từng claim
            if (data.details && data.details.length > 0) {
                let html = "<b>🔎 Chi tiết kiểm chứng:</b><ul>";
                data.details.forEach(d => {
                    const icon = d.status === 'REFUTED' ? '❌' : (d.status === 'SUPPORTED' ? '✅' : '⚪');
                    html += `<li style="margin-bottom: 5px;">${icon} ${d.claim}</li>`;
                });
                html += "</ul>";
                detailsDiv.innerHTML += html;
            }

        } catch (err) {
            statusDiv.textContent = "❌ Lỗi kết nối Server: " + err.message;
        } finally {
            btn.disabled = false;
        }
    });
});

// Hàm này sẽ chạy trực tiếp trên trang web (Content Script)
function getPageContent() {
    // Logic lấy tin riêng cho VnExpress (hoặc các trang báo chung)
    const title = document.querySelector('h1.title-detail')?.innerText || document.title;
    const content = document.querySelector('article.fck_detail')?.innerText || document.body.innerText;
    
    // Cắt bớt nội dung nếu quá dài để gửi API cho nhanh (Model cũng chỉ cần đoạn đầu)
    return {
        title: title,
        content: content.substring(0, 3000) // Lấy 3000 ký tự đầu
    };
}
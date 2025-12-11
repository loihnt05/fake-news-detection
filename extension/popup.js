document.addEventListener('DOMContentLoaded', function() {
    const input = document.getElementById('newsInput');
    const btn = document.getElementById('checkBtn');
    const resultBox = document.getElementById('result-box');
    const label = document.getElementById('resLabel');
    const msg = document.getElementById('resMsg');
    const loader = document.getElementById('loader');

    // 1. Tự động lấy text đang bôi đen trên web
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
        chrome.scripting.executeScript({
            target: {tabId: tabs[0].id},
            function: () => window.getSelection().toString()
        }, (results) => {
            if (results && results[0] && results[0].result) {
                input.value = results[0].result; // Điền vào ô input
            }
        });
    });

    // 2. Xử lý khi bấm nút Kiểm tra
    btn.addEventListener('click', async () => {
        const text = input.value.trim();
        if (!text) return alert("Vui lòng nhập nội dung!");

        // Reset giao diện
        btn.disabled = true;
        loader.style.display = 'block';
        resultBox.style.display = 'none';

        try {
            // GỌI API SERVER PYTHON
            const response = await fetch('http://127.0.0.1:8000/check-news', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            });

            const data = await response.json();

            // Hiển thị kết quả
            resultBox.style.display = 'block';
            resultBox.className = ''; // Xóa class cũ
            
            // Logic màu sắc
            if (data.result === 'REAL') {
                resultBox.classList.add('real');
                label.innerText = "✅ TIN THẬT";
            } else if (data.result === 'FAKE') {
                resultBox.classList.add('fake');
                label.innerText = "⚠️ TIN GIẢ";
            } else {
                resultBox.classList.add('undefined');
                label.innerText = "🤔 CHƯA RÕ";
            }

            msg.innerText = `${data.message}\n(Độ tin cậy: ${(data.confidence * 100).toFixed(1)}%)`;

        } catch (error) {
            alert("❌ Lỗi kết nối Server! Bạn đã chạy 'uv run uvicorn main:app' chưa?");
            console.error(error);
        } finally {
            btn.disabled = false;
            loader.style.display = 'none';
        }
    });
});
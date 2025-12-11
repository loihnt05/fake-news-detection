import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. CẤU HÌNH (Sửa tên file/cột của bạn ở đây)
# ==========================================
INPUT_FILE = 'articles.csv'  # Tên file dữ liệu gốc của bạn
OUTPUT_FILE = 'articles_clean.csv' # Tên file sau khi làm sạch
COL_TEXT = 'content'   # Tên cột chứa nội dung bài báo
COL_LABEL = 'label' # Tên cột chứa nhãn (0, 1 hoặc Real, Fake)

# ==========================================
# 2. LOAD DỮ LIỆU
# ==========================================
print(f"⏳ Đang đọc file {INPUT_FILE}...")
try:
    df = pd.read_csv(INPUT_FILE)
    # Nếu file là Excel thì dùng: df = pd.read_excel(INPUT_FILE)
    print(f"✅ Đã load xong. Tổng số dòng ban đầu: {len(df):,}")
except Exception as e:
    print(f"❌ Lỗi không đọc được file: {e}")
    exit()

# ==========================================
# 3. THỐNG KÊ TỶ LỆ (Trước khi xóa)
# ==========================================
print("\n--- 📊 THỐNG KÊ BAN ĐẦU ---")
count = df[COL_LABEL].value_counts()
percent = df[COL_LABEL].value_counts(normalize=True) * 100

print(f"Số lượng từng nhãn:\n{count}")
print(f"Tỷ lệ phần trăm:\n{percent}")

# Vẽ biểu đồ tròn (Optional - để đưa vào báo cáo)
plt.figure(figsize=(6,6))
count.plot.pie(autopct='%.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
plt.title('Tỷ lệ Real/Fake ban đầu')
plt.ylabel('')
plt.show() # Tắt dòng này nếu chạy trên server không màn hình

# ==========================================
# 4. KIỂM TRA & XỬ LÝ TRÙNG LẶP
# ==========================================
print("\n--- 🧹 ĐANG LÀM SẠCH ---")

# Kiểm tra xem có bài nào nội dung giống hệt nhau không
duplicates = df.duplicated(subset=[COL_TEXT], keep='first')
num_duplicates = duplicates.sum()
print(f"⚠️ Phát hiện {num_duplicates:,} bài báo bị trùng nội dung.")

# Kiểm tra MÂU THUẪN (Cùng nội dung nhưng khác nhãn) -> Cái này rất hại model
# Group theo text và đếm số lượng nhãn unique
conflict_check = df.groupby(COL_TEXT)[COL_LABEL].nunique()
conflicts = conflict_check[conflict_check > 1]

if len(conflicts) > 0:
    print(f"⛔ CẢNH BÁO ĐỎ: Có {len(conflicts)} bài viết bị gán SAI NHÃN (vừa là Real vừa là Fake).")
    print("   -> Hệ thống sẽ xóa toàn bộ các bài mâu thuẫn này để tránh làm model bị 'điên'.")
    # Lấy danh sách text bị mâu thuẫn
    bad_texts = conflicts.index.tolist()
    # Xóa những dòng chứa text này
    df = df[~df[COL_TEXT].isin(bad_texts)]
else:
    print("✅ Kiểm tra an toàn: Không có bài viết nào bị xung đột nhãn.")

# Xóa trùng lặp thông thường (Giữ lại bản ghi đầu tiên)
df_clean = df.drop_duplicates(subset=[COL_TEXT], keep='first')

# ==========================================
# 5. KẾT QUẢ & LƯU FILE
# ==========================================
print("\n--- 🏁 KẾT QUẢ SAU KHI LỌC ---")
print(f"Dữ liệu gốc:     {len(df):,} dòng")
print(f"Dữ liệu sạch:    {len(df_clean):,} dòng")
print(f"Đã loại bỏ:      {len(df) - len(df_clean):,} dòng rác")

# Thống kê lại tỷ lệ mới
print("\nTỷ lệ sau khi làm sạch:")
print(df_clean[COL_LABEL].value_counts(normalize=True) * 100)

# Lưu ra file mới
df_clean.to_csv(OUTPUT_FILE, index=False)
print(f"\n💾 Đã lưu file sạch vào: {OUTPUT_FILE}")
print("👉 Hãy dùng file này cho bước Vector Database tiếp theo!")
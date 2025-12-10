#!/usr/bin/env python3
"""
Quick launcher cho Jupyter Notebook training
"""
import subprocess
import sys
import os

print("=" * 60)
print("JUPYTER NOTEBOOK LAUNCHER")
print("=" * 60)

# Check if we're in the right directory
if not os.path.exists('train_classifier.ipynb'):
    print("\n❌ Không tìm thấy train_classifier.ipynb")
    print("💡 Chạy script này trong thư mục model/")
    sys.exit(1)

print("\n📝 Các tùy chọn:")
print("1. Jupyter Notebook (Classic)")
print("2. JupyterLab (Modern)")
print("3. VS Code (mở file)")
print("4. Kiểm tra setup trước")
print("5. Thoát")

choice = input("\nChọn (1-5): ").strip()

if choice == "1":
    print("\n🚀 Đang khởi động Jupyter Notebook...")
    print("💡 Notebook sẽ mở trong browser")
    print("💡 Nhấn Ctrl+C để tắt server\n")
    subprocess.run(["jupyter", "notebook", "train_classifier.ipynb"])

elif choice == "2":
    print("\n🚀 Đang khởi động JupyterLab...")
    print("💡 JupyterLab sẽ mở trong browser")
    print("💡 Nhấn Ctrl+C để tắt server\n")
    subprocess.run(["jupyter", "lab", "train_classifier.ipynb"])

elif choice == "3":
    print("\n🚀 Đang mở VS Code...")
    subprocess.run(["code", "train_classifier.ipynb"])
    print("\n✅ Đã mở file trong VS Code")
    print("💡 Chọn kernel Python và Run All Cells")

elif choice == "4":
    print("\n🔍 Đang kiểm tra setup...\n")
    subprocess.run([sys.executable, "test_notebook_setup.py"])

elif choice == "5":
    print("\n👋 Tạm biệt!")
    sys.exit(0)

else:
    print("\n❌ Lựa chọn không hợp lệ!")
    sys.exit(1)

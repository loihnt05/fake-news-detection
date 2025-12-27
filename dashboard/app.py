import streamlit as st
import pandas as pd
import psycopg2
import requests
import os
import plotly.express as px
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- CẤU HÌNH ---
st.set_page_config(page_title="Fact-Check Admin", layout="wide", page_icon="🛡️")

DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB", "vnexpress_scraper"),
    "user": os.getenv("POSTGRES_USER", "admin"),
    "password": os.getenv("POSTGRES_PASSWORD", "admin"),
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432")
}

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_URL = f"{BACKEND_URL}/api/v1"

# --- HELPER FUNCTIONS ---
def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

def load_data(query):
    conn = get_db_connection()
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def approve_report(report_id, verdict):
    """Gọi API Backend để duyệt report (Cập nhật Reputation)"""
    try:
        resp = requests.post(f"{API_URL}/admin/approve-report", json={
            "report_id": report_id,
            "verdict": verdict # 'APPROVED' hoặc 'REJECTED'
        })
        if resp.status_code == 200:
            st.success(f"Đã xử lý: {verdict}")
            st.rerun()
        else:
            st.error(f"Lỗi API: {resp.text}")
    except Exception as e:
        st.error(f"Lỗi kết nối: {e}")

# --- GIAO DIỆN CHÍNH ---
st.title("🛡️ Hệ thống Quản trị Fact-Check AI")

# Sidebar Menu
menu = st.sidebar.radio("Menu", ["📊 Tổng quan", "📨 Duyệt Báo Cáo (Review)", "👥 Quản lý User"])

# TAB 1: TỔNG QUAN
if menu == "📊 Tổng quan":
    col1, col2, col3 = st.columns(3)
    
    with col1:
        count = load_data("SELECT COUNT(*) FROM claims WHERE system_label='REAL'").iloc[0,0]
        st.metric("Tin đã xác thực (REAL)", count)
        
    with col2:
        count = load_data("SELECT COUNT(*) FROM user_reports WHERE status='PENDING'").iloc[0,0]
        st.metric("Báo cáo chờ duyệt", count, delta_color="inverse")
        
    with col3:
        count = load_data("SELECT COUNT(*) FROM users").iloc[0,0]
        st.metric("Tổng Users", count)

    st.markdown("---")
    st.subheader("📈 Xu hướng báo cáo")
    
    # Biểu đồ Realtime
    df_trend = load_data("""
        SELECT DATE(created_at) as date, user_feedback, COUNT(*) as count 
        FROM user_reports 
        GROUP BY 1, 2 ORDER BY 1
    """)
    if not df_trend.empty:
        fig = px.bar(df_trend, x="date", y="count", color="user_feedback", title="Số lượng Report theo ngày")
        st.plotly_chart(fig, use_container_width=True)

# TAB 2: DUYỆT BÁO CÁO (QUAN TRỌNG NHẤT)
elif menu == "📨 Duyệt Báo Cáo (Review)":
    st.header("Danh sách báo cáo chờ xử lý")
    
    # Lấy danh sách Pending, Join với Claims để hiện nội dung gốc
    df_pending = load_data("""
        SELECT r.id, r.user_feedback, r.comment, r.ai_label_at_report, r.ai_confidence, 
               r.model_version, c.content as claim_content, u.reputation_score
        FROM user_reports r
        JOIN claims c ON r.claim_id = c.id
        LEFT JOIN users u ON r.user_id = u.id
        WHERE r.status = 'PENDING'
        ORDER BY r.created_at ASC
    """)
    
    if df_pending.empty:
        st.info("Tuyệt vời! Không có báo cáo nào cần xử lý.")
    else:
        for index, row in df_pending.iterrows():
            with st.expander(f"{row['user_feedback']} | {row['claim_content'][:80]}...", expanded=True):
                c1, c2 = st.columns([2, 1])
                
                with c1:
                    st.markdown(f"**Nội dung Claim:**")
                    st.info(row['claim_content'])
                    st.markdown(f"**User Comment:** `{row['comment']}`")
                    
                    # So sánh AI vs User
                    st.markdown("#### ⚔️ Đối đầu:")
                    col_ai, col_user = st.columns(2)
                    col_ai.warning(f"🤖 AI nói: {row['ai_label_at_report']} ({row['ai_confidence']:.2f})")
                    col_user.error(f"👤 User nói: {row['user_feedback']}")

                with c2:
                    st.markdown("**Thông tin User:**")
                    st.progress(row['reputation_score'], text=f"Uy tín: {row['reputation_score']}")
                    st.caption(f"Model Version: {row['model_version']}")
                    
                    st.markdown("---")
                    # Hành động
                    btn_col1, btn_col2 = st.columns(2)
                    if btn_col1.button("✅ DUYỆT (Đúng)", key=f"app_{row['id']}"):
                        approve_report(row['id'], 'APPROVED')
                        
                    if btn_col2.button("❌ BÁC BỎ (Sai)", key=f"rej_{row['id']}"):
                        approve_report(row['id'], 'REJECTED')

# TAB 3: USER MANAGER
elif menu == "👥 Quản lý User":
    st.subheader("Top Users đóng góp tích cực")
    df_users = load_data("""
        SELECT id, role, reputation_score, total_reports, accepted_reports, last_active_at
        FROM users
        ORDER BY reputation_score DESC
        LIMIT 20
    """)
    st.dataframe(df_users, use_container_width=True)
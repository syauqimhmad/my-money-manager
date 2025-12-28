import streamlit as st
from st_supabase_connection import SupabaseConnection
import pandas as pd
import plotly.express as px

# 1. Konfigurasi Halaman
st.set_page_config(page_title="AI Money Manager", page_icon="💰", layout="wide")

# 2. Koneksi ke Database (Otomatis baca Secrets)
conn = st.connection("supabase", type=SupabaseConnection)

# 3. Fungsi Ambil Data (Cache 1 menit agar tidak berat)
@st.cache_data(ttl=60)
def load_data():
    # Ambil semua data dari tabel transactions
    # Urutkan dari yang terbaru
    response = conn.table("transactions").select("*").order("date", desc=True).execute()
    return response.data

# --- UI UTAMA ---
st.title("🤖 AI Financial Dashboard")

# Load Data
rows = load_data()

if not rows:
    st.info("Belum ada data transaksi. Silakan chat bot Telegram Anda!")
else:
    # Konversi ke Pandas DataFrame agar mudah diolah
    df = pd.DataFrame(rows)
    
    # Format ulang kolom tanggal
    df['date'] = pd.to_datetime(df['date']).dt.date

    # --- BAGIAN 1: RINGKASAN (KPI) ---
    st.subheader("Ringkasan Bulan Ini")
    col1, col2, col3 = st.columns(3)
    
    # Hitung Pemasukan & Pengeluaran
    pemasukan = df[df['amount'] > 0]['amount'].sum()
    pengeluaran = df[df['amount'] < 0]['amount'].sum()
    sisa = pemasukan + pengeluaran

    col1.metric("Pemasukan", f"Rp {pemasukan:,.0f}")
    col2.metric("Pengeluaran", f"Rp {abs(pengeluaran):,.0f}", delta_color="inverse")
    col3.metric("Sisa Saldo", f"Rp {sisa:,.0f}")

    st.divider()

    # --- BAGIAN 2: GRAFIK VISUAL ---
    c1, c2 = st.columns((2, 1))
    
    with c1:
        st.subheader(" tren Pengeluaran Harian")
        # Filter hanya pengeluaran (angka negatif)
        df_expense = df[df['amount'] < 0].copy()
        df_expense['amount'] = df_expense['amount'].abs() # Ubah jadi positif biar grafik naik
        
        # Group by tanggal
        daily_spend = df_expense.groupby('date')['amount'].sum().reset_index()
        
        fig_line = px.bar(daily_spend, x='date', y='amount', title="Pengeluaran per Hari")
        st.plotly_chart(fig_line, use_container_width=True)

    with c2:
        st.subheader("Kategori Terbesar")
        # Group by kategori
        cat_spend = df_expense.groupby('category')['amount'].sum().reset_index()
        fig_pie = px.pie(cat_spend, values='amount', names='category', hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

    # --- BAGIAN 3: TABEL DETAIL ---
    st.subheader("Riwayat Transaksi Terakhir")
    
    # Tampilkan kolom yang relevan saja
    display_cols = ['date', 'shop', 'description', 'category', 'amount', 'qty', 'uom']
    
    # Pastikan kolom ada (jaga-jaga jika data lama belum punya kolom shop)
    available_cols = [c for c in display_cols if c in df.columns]
    
    st.dataframe(
        df[available_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "amount": st.column_config.NumberColumn(
                "Nominal", format="Rp %d"
            ),
            "date": "Tanggal",
            "shop": "Toko/Merchant",
            "description": "Item",
            "category": "Kategori"
        }
    )
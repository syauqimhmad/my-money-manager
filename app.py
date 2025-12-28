import streamlit as st
from st_supabase_connection import SupabaseConnection
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# Konfigurasi Halaman
st.set_page_config(page_title="AI Money Manager", page_icon="💰", layout="wide")

# Koneksi ke Database
conn = st.connection("supabase", type=SupabaseConnection)

# Fungsi Ambil Data
@st.cache_data(ttl=60)
def load_data():
    response = conn.table("transactions").select("*").order("date", desc=False).execute()
    return response.data

# Fungsi Helper untuk Perhitungan
def calculate_metrics(df, start_date, end_date):
    # Create a copy and extract date part only
    df_copy = df.copy()
    
    # Convert to date strings for safe comparison
    df_copy['date_str'] = pd.to_datetime(df_copy['date']).dt.date
    
    # Filter by date range
    df_period = df_copy[(df_copy['date_str'] >= start_date) & (df_copy['date_str'] <= end_date)].copy()
    
    # Restore original date column
    df_period['date'] = pd.to_datetime(df_period['date'])
    
    income = df_period[df_period['amount'] > 0]['amount'].sum()
    expense = abs(df_period[df_period['amount'] < 0]['amount'].sum())
    net = income - expense
    savings_rate = (net / income * 100) if income > 0 else 0
    return income, expense, net, savings_rate, df_period

def get_previous_period(start_date, end_date):
    delta = end_date - start_date
    prev_end = start_date - timedelta(days=1)
    prev_start = prev_end - delta
    return prev_start, prev_end

# Load Data
rows = load_data()

if not rows:
    st.info("Belum ada data transaksi. Silakan chat bot Telegram Anda!")
    st.stop()

# Prepare DataFrame
df = pd.DataFrame(rows)

# Convert date column to datetime, handling various formats
try:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['date_only'] = df['date'].dt.date  # Create date-only column for filtering
except:
    st.error("Error converting date column. Please check your data format.")
    st.stop()

# Remove rows with invalid dates
df = df.dropna(subset=['date'])

if df.empty:
    st.info("No valid transactions found. Please check your data.")
    st.stop()

df['year_month'] = df['date'].dt.to_period('M')
df['month_name'] = df['date'].dt.strftime('%B %Y')

# --- HEADER & FILTERS ---
st.title("🤖 AI Financial Management Dashboard")

col_filter1, col_filter2 = st.columns([3, 1])

with col_filter1:
    period_option = st.selectbox(
        "Pilih Periode",
        ["Bulan Ini", "Bulan Lalu", "3 Bulan Terakhir", "6 Bulan Terakhir", "Tahun Ini", "Custom"],
        index=0
    )

today = datetime.now().date()
first_day_month = today.replace(day=1)

if period_option == "Bulan Ini":
    start_date = first_day_month
    end_date = today
elif period_option == "Bulan Lalu":
    start_date = (first_day_month - relativedelta(months=1))
    end_date = first_day_month - timedelta(days=1)
elif period_option == "3 Bulan Terakhir":
    start_date = first_day_month - relativedelta(months=2)
    end_date = today
elif period_option == "6 Bulan Terakhir":
    start_date = first_day_month - relativedelta(months=5)
    end_date = today
elif period_option == "Tahun Ini":
    start_date = today.replace(month=1, day=1)
    end_date = today
else:  # Custom
    with col_filter2:
        date_range = st.date_input(
            "Pilih Range",
            value=(first_day_month, today),
            max_value=today
        )
        if len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date, end_date = first_day_month, today

# Calculate Current Period
income, expense, net, savings_rate, df_period = calculate_metrics(df, start_date, end_date)

# Calculate Previous Period for Comparison
prev_start, prev_end = get_previous_period(start_date, end_date)
prev_income, prev_expense, prev_net, prev_savings_rate, _ = calculate_metrics(df, prev_start, prev_end)

# --- KPI METRICS ---
st.subheader(f"📊 Ringkasan Periode: {start_date.strftime('%d %b %Y')} - {end_date.strftime('%d %b %Y')}")

col1, col2, col3, col4, col5 = st.columns(5)

income_delta = ((income - prev_income) / prev_income * 100) if prev_income > 0 else 0
expense_delta = ((expense - prev_expense) / prev_expense * 100) if prev_expense > 0 else 0
net_delta = ((net - prev_net) / prev_net * 100) if prev_net != 0 else 0

col1.metric("💰 Pemasukan", f"Rp {income:,.0f}", f"{income_delta:+.1f}%")
col2.metric("💸 Pengeluaran", f"Rp {expense:,.0f}", f"{expense_delta:+.1f}%", delta_color="inverse")
col3.metric("💵 Net Cash Flow", f"Rp {net:,.0f}", f"{net_delta:+.1f}%")
col4.metric("📈 Savings Rate", f"{savings_rate:.1f}%", f"{savings_rate - prev_savings_rate:+.1f}%")

# Financial Health Score
health_score = min(100, max(0, (savings_rate * 0.5) + (50 if net > 0 else 0)))
health_status = "Excellent" if health_score >= 80 else "Good" if health_score >= 60 else "Fair" if health_score >= 40 else "Poor"
col5.metric("💪 Health Score", f"{health_score:.0f}/100", health_status)

st.divider()

# --- TABS FOR DIFFERENT ANALYSES ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Overview", "🏷️ Categories", "💳 Merchants", "📅 Trends", "📊 Details"
])

# TAB 1: OVERVIEW
with tab1:
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Cash Flow Trend")
        
        # Daily cash flow
        df_period_copy = df_period.copy()
        daily_flow = df_period_copy.groupby(df_period_copy['date'].dt.date).agg({
            'amount': 'sum'
        }).reset_index()
        daily_flow.columns = ['date', 'amount']
        daily_flow['cumulative'] = daily_flow['amount'].cumsum()
        
        fig_flow = go.Figure()
        fig_flow.add_trace(go.Bar(
            x=daily_flow['date'],
            y=daily_flow['amount'],
            name='Daily Flow',
            marker_color=['green' if x > 0 else 'red' for x in daily_flow['amount']]
        ))
        fig_flow.add_trace(go.Scatter(
            x=daily_flow['date'],
            y=daily_flow['cumulative'],
            name='Cumulative',
            mode='lines',
            line=dict(color='blue', width=2)
        ))
        fig_flow.update_layout(height=350, showlegend=True)
        st.plotly_chart(fig_flow, use_container_width=True)
    
    with col_b:
        st.subheader("Income vs Expense Breakdown")
        
        # Pie chart for income vs expense
        breakdown_data = pd.DataFrame({
            'Type': ['Income', 'Expense', 'Net Savings'],
            'Amount': [income, expense, net if net > 0 else 0]
        })
        
        fig_pie = px.pie(
            breakdown_data,
            values='Amount',
            names='Type',
            hole=0.4,
            color='Type',
            color_discrete_map={'Income': '#10b981', 'Expense': '#ef4444', 'Net Savings': '#3b82f6'}
        )
        fig_pie.update_layout(height=350)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Top Transactions
    st.subheader("🔥 Top 10 Transactions")
    df_period_sorted = df_period.copy()
    df_period_sorted['amount_abs'] = df_period_sorted['amount'].abs()
    
    # Ensure columns exist before selecting
    available_cols = ['date', 'description', 'category', 'amount', 'qty', 'uom']
    if 'shop' in df_period_sorted.columns:
        available_cols.insert(1, 'shop')
    
    display_cols_top = [c for c in available_cols if c in df_period_sorted.columns]
    
    top_transactions = df_period_sorted.nlargest(10, 'amount_abs')[display_cols_top]
    st.dataframe(
        top_transactions,
        use_container_width=True,
        hide_index=True,
        column_config={
            "amount": st.column_config.NumberColumn("Nominal", format="Rp %.0f"),
            "date": "Tanggal"
        }
    )

# TAB 2: CATEGORIES
with tab2:
    st.subheader("📊 Category Analysis")
    
    # Expense by category
    df_expense = df_period[df_period['amount'] < 0].copy()
    df_expense['amount_abs'] = df_expense['amount'].abs()
    
    if not df_expense.empty and 'category' in df_expense.columns:
        col_cat1, col_cat2 = st.columns([2, 1])
        
        with col_cat1:
            cat_expense = df_expense.groupby('category')['amount_abs'].sum().reset_index()
            cat_expense = cat_expense.sort_values('amount_abs', ascending=False)
            
            fig_cat_bar = px.bar(
                cat_expense,
                x='amount_abs',
                y='category',
                orientation='h',
                title="Pengeluaran per Kategori",
                labels={'amount_abs': 'Jumlah (Rp)', 'category': 'Kategori'},
                color='amount_abs',
                color_continuous_scale='Reds'
            )
            fig_cat_bar.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_cat_bar, use_container_width=True)
        
        with col_cat2:
            fig_cat_pie = px.pie(
                cat_expense,
                values='amount_abs',
                names='category',
                title="Proporsi Kategori"
            )
            fig_cat_pie.update_layout(height=400)
            st.plotly_chart(fig_cat_pie, use_container_width=True)
        
        # Category Details
        st.subheader("Detail per Kategori")
        for category in cat_expense['category'].head(5):
            with st.expander(f"📁 {category} - Rp {cat_expense[cat_expense['category']==category]['amount_abs'].values[0]:,.0f}"):
                detail_cols = ['date', 'description', 'amount', 'qty', 'uom']
                if 'shop' in df_expense.columns:
                    detail_cols.insert(2, 'shop')
                detail_cols_available = [c for c in detail_cols if c in df_expense.columns]
                
                cat_data = df_expense[df_expense['category'] == category][detail_cols_available].sort_values('date', ascending=False)
                st.dataframe(cat_data, hide_index=True, use_container_width=True)
    else:
        st.info("Tidak ada data pengeluaran untuk periode ini")

# TAB 3: MERCHANTS
with tab3:
    st.subheader("🏪 Merchant Analysis")
    
    df_expense = df_period[df_period['amount'] < 0].copy()
    df_expense['amount_abs'] = df_expense['amount'].abs()
    
    if not df_expense.empty and 'shop' in df_expense.columns:
        # Top merchants by spending
        merchant_spend = df_expense.groupby('shop').agg({
            'amount_abs': 'sum',
            'description': 'count'
        }).reset_index()
        merchant_spend.columns = ['Merchant', 'Total Spending', 'Transaction Count']
        merchant_spend = merchant_spend.sort_values('Total Spending', ascending=False).head(15)
        
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            fig_merchant = px.bar(
                merchant_spend,
                x='Total Spending',
                y='Merchant',
                orientation='h',
                title="Top 15 Merchants by Spending",
                color='Total Spending',
                color_continuous_scale='Blues'
            )
            fig_merchant.update_layout(height=500)
            st.plotly_chart(fig_merchant, use_container_width=True)
        
        with col_m2:
            fig_merchant_count = px.bar(
                merchant_spend,
                x='Transaction Count',
                y='Merchant',
                orientation='h',
                title="Top 15 Merchants by Frequency",
                color='Transaction Count',
                color_continuous_scale='Greens'
            )
            fig_merchant_count.update_layout(height=500)
            st.plotly_chart(fig_merchant_count, use_container_width=True)
        
        # Merchant details table
        st.dataframe(
            merchant_spend,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Total Spending": st.column_config.NumberColumn("Total", format="Rp %.0f")
            }
        )
    else:
        st.info("Data merchant tidak tersedia")

# TAB 4: TRENDS
with tab4:
    st.subheader("📅 Trend Analysis")
    
    # Monthly trend (last 6 months)
    six_months_ago = first_day_month - relativedelta(months=5)
    
    # Use date_only column for filtering
    df_trend = df[df['date_only'] >= six_months_ago].copy()
    
    if not df_trend.empty:
        monthly_summary = df_trend.groupby('year_month').apply(
            lambda x: pd.Series({
                'Income': x[x['amount'] > 0]['amount'].sum(),
                'Expense': abs(x[x['amount'] < 0]['amount'].sum()),
                'Net': x['amount'].sum()
            }), include_groups=False
        ).reset_index()
        monthly_summary['month_str'] = monthly_summary['year_month'].astype(str)
        
        # Income vs Expense Trend
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=monthly_summary['month_str'],
            y=monthly_summary['Income'],
            name='Income',
            mode='lines+markers',
            line=dict(color='green', width=3)
        ))
        fig_trend.add_trace(go.Scatter(
            x=monthly_summary['month_str'],
            y=monthly_summary['Expense'],
            name='Expense',
            mode='lines+markers',
            line=dict(color='red', width=3)
        ))
        fig_trend.add_trace(go.Scatter(
            x=monthly_summary['month_str'],
            y=monthly_summary['Net'],
            name='Net',
            mode='lines+markers',
            line=dict(color='blue', width=2, dash='dash')
        ))
        fig_trend.update_layout(
            title="6-Month Income vs Expense Trend",
            xaxis_title="Month",
            yaxis_title="Amount (Rp)",
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Category trend over time
        df_cat_trend_data = df_trend[df_trend['amount'] < 0].copy()
        if not df_cat_trend_data.empty and 'category' in df_cat_trend_data.columns:
            st.subheader("Expense Category Trends")
            
            df_cat_trend_data['amount_abs'] = df_cat_trend_data['amount'].abs()
            
            cat_monthly = df_cat_trend_data.groupby(['year_month', 'category'])['amount_abs'].sum().reset_index()
            cat_monthly['month_str'] = cat_monthly['year_month'].astype(str)
            
            # Get top 5 categories
            top_cats = df_cat_trend_data.groupby('category')['amount_abs'].sum().nlargest(5).index
            cat_monthly_top = cat_monthly[cat_monthly['category'].isin(top_cats)]
            
            if not cat_monthly_top.empty:
                fig_cat_trend = px.line(
                    cat_monthly_top,
                    x='month_str',
                    y='amount_abs',
                    color='category',
                    title="Top 5 Categories Expense Trend",
                    markers=True
                )
                fig_cat_trend.update_layout(height=400)
                st.plotly_chart(fig_cat_trend, use_container_width=True)
    else:
        st.info("Insufficient data for trend analysis")

# TAB 5: DETAILS
with tab5:
    st.subheader("📋 All Transactions")
    
    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        trans_type = st.selectbox(
            "Transaction Type",
            ["All", "Income", "Expense"],
            index=0
        )
    
    with col_f2:
        if 'category' in df_period.columns:
            categories = ["All"] + sorted(df_period['category'].dropna().unique().tolist())
            selected_cat = st.selectbox("Category", categories)
        else:
            selected_cat = "All"
    
    with col_f3:
        if 'shop' in df_period.columns:
            shops = ["All"] + sorted(df_period['shop'].dropna().unique().tolist())
            selected_shop = st.selectbox("Merchant", shops)
        else:
            selected_shop = "All"
    
    # Apply filters
    df_filtered = df_period.copy()
    
    if trans_type == "Income":
        df_filtered = df_filtered[df_filtered['amount'] > 0]
    elif trans_type == "Expense":
        df_filtered = df_filtered[df_filtered['amount'] < 0]
    
    if selected_cat != "All" and 'category' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['category'] == selected_cat]
    
    if selected_shop != "All" and 'shop' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['shop'] == selected_shop]
    
    # Summary of filtered data
    filtered_total = df_filtered['amount'].sum()
    filtered_count = len(df_filtered)
    
    st.info(f"Showing {filtered_count} transactions | Total: Rp {filtered_total:,.0f}")
    
    # Display table
    display_cols = ['date', 'description', 'category', 'amount', 'qty', 'uom']
    if 'shop' in df_filtered.columns:
        display_cols.insert(1, 'shop')
    
    available_cols = [c for c in display_cols if c in df_filtered.columns]
    
    df_display = df_filtered[available_cols].sort_values('date', ascending=False)
    
    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "amount": st.column_config.NumberColumn("Nominal", format="Rp %.0f"),
            "date": "Tanggal",
            "shop": "Merchant",
            "description": "Description",
            "category": "Category"
        },
        height=600
    )
    
    # Download button
    csv = df_display.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"transactions_{start_date}_{end_date}.csv",
        mime="text/csv"
    )

# Footer
st.divider()
st.caption(f"Last updated: {datetime.now().strftime('%d %B %Y, %H:%M:%S')} | Total Transactions: {len(df)}")
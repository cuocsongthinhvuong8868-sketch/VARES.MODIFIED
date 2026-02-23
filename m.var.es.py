import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.stats import norm
from matplotlib.backends.backend_pdf import PdfPages
import datetime
import tempfile

# ==============================================================================
# 1. CẤU HÌNH THÔNG SỐ VÀ DANH MỤC
# ==============================================================================
st.set_page_config(page_title="Hệ Thống Giám Sát Rủi Ro Vĩ Mô", layout="wide")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']

WINDOW_YEARS = 3
TRADING_DAYS = 252
WINDOW_SIZE = WINDOW_YEARS * TRADING_DAYS
MIN_PERIODS = 252 
CONFIDENCE = 0.95

STRESS_THRESHOLD_VN30 = 0.40       
COMPLACENCY_THRESHOLD_MKT = 0.50   

VN30_TICKERS = [
    'ACB.VN', 'BCM.VN', 'BID.VN', 'BVH.VN', 'CTG.VN', 'FPT.VN', 'GAS.VN', 'GVR.VN', 'HDB.VN', 'HPG.VN',
    'MBB.VN', 'MSN.VN', 'MWG.VN', 'PLX.VN', 'POW.VN', 'SAB.VN', 'SHB.VN', 'SSB.VN', 'SSI.VN', 'STB.VN',
    'TCB.VN', 'TPB.VN', 'VCB.VN', 'VHM.VN', 'VIB.VN', 'VIC.VN', 'VJC.VN', 'VNM.VN', 'VPB.VN', 'VRE.VN'
]

MARKET_TICKERS = {
    'Ngân Hàng': ['VCB.VN', 'BID.VN', 'CTG.VN', 'MBB.VN', 'TCB.VN', 'VPB.VN', 'ACB.VN', 'STB.VN', 'HDB.VN', 'VIB.VN'],
    'Bất Động Sản': ['VHM.VN', 'NVL.VN', 'DIG.VN', 'DXG.VN', 'KDH.VN', 'NLG.VN', 'PDR.VN'],
    'Chứng Khoán': ['SSI.VN', 'VND.VN', 'VCI.VN', 'HCM.VN', 'FTS.VN', 'BSI.VN', 'VIX.VN'], 
    'Thép / Vật Liệu': ['HPG.VN', 'HSG.VN', 'NKG.VN', 'HT1.VN', 'BCC.VN'],
    'Dầu Khí': ['GAS.VN', 'PVD.VN', 'CNG.VN', 'BSR.VN', 'PLX.VN', 'VIP.VN', 'VTO.VN'], 
    'Bán Lẻ': ['MWG.VN', 'PNJ.VN', 'FRT.VN', 'DGW.VN'],
    'Khu Công Nghiệp': ['BCM.VN', 'IDC.VN', 'KBC.VN', 'SZC.VN', 'VGC.VN'],
    'Công Nghệ': ['FPT.VN', 'CMG.VN', 'VGI.VN'],
    'Cảng Biển / Logistics': ['GMD.VN', 'HAH.VN', 'VSC.VN'],
    'Nông Nghiệp / Thủy Sản': ['VHC.VN', 'ANV.VN', 'DBC.VN', 'HAG.VN'],
    'Tiện Ích': ['POW.VN', 'REE.VN', 'NT2.VN', 'GEG.VN']
}

ALL_MARKET_TICKERS = [ticker for sublist in MARKET_TICKERS.values() for ticker in sublist]

# ==============================================================================
# 2. CÁC HÀM TÍNH TOÁN LÕI 
# ==============================================================================
@st.cache_data(ttl=3600)
def fetch_data(tickers, start_date, end_date):
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        if data.empty: return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.levels[0]: df = data.xs('Close', level=0, axis=1)
            elif 'Adj Close' in data.columns.levels[0]: df = data.xs('Adj Close', level=0, axis=1)
            else: return pd.DataFrame()
        else:
            if 'Close' in data.columns: df = data[['Close']].copy(); df.columns = [tickers[0]]
            else: return pd.DataFrame()
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        return df.dropna(how='all')
    except Exception as e:
        return pd.DataFrame()

def calculate_es_robust(returns, window, min_periods, confidence):
    returns_arr = np.nan_to_num(returns, nan=0.0)
    n = len(returns_arr)
    es_values = np.full(n, np.nan)
    if n <= min_periods: return es_values
    for i in range(min_periods, n):
        start_idx = max(0, i - window)
        window_data = returns_arr[start_idx:i]
        var_t = np.percentile(window_data, (1 - confidence) * 100)
        tails = window_data[window_data <= var_t]
        es_values[i] = np.mean(tails) if len(tails) > 0 else var_t
    return es_values

def calculate_risk_metrics(df_price):
    df_return = df_price.pct_change()
    rolling_mean = df_return.rolling(window=WINDOW_SIZE, min_periods=MIN_PERIODS).mean()
    rolling_std = df_return.rolling(window=WINDOW_SIZE, min_periods=MIN_PERIODS).std()
    rolling_skew = df_return.rolling(window=WINDOW_SIZE, min_periods=MIN_PERIODS).skew().fillna(0)
    rolling_kurt = df_return.rolling(window=WINDOW_SIZE, min_periods=MIN_PERIODS).kurt().fillna(0)
    
    z_score = norm.ppf(1 - CONFIDENCE)
    z_cf = (z_score + (z_score**2 - 1) * rolling_skew / 6 + (z_score**3 - 3 * z_score) * rolling_kurt / 24 - (2 * z_score**3 - 5 * z_score) * (rolling_skew**2) / 36)
    df_var = rolling_mean + z_cf * rolling_std
    
    df_es = pd.DataFrame(index=df_return.index, columns=df_return.columns)
    for col in df_return.columns:
        df_es[col] = calculate_es_robust(df_return[col].values, WINDOW_SIZE, MIN_PERIODS, CONFIDENCE)
        
    df_spread = df_var - df_es 
    return df_return, df_var, df_es, df_spread

# ==============================================================================
# 3. GIAO DIỆN STREAMLIT
# ==============================================================================
st.title("Hệ Thống Phân Tích Rủi Ro Vĩ Mô & Định Giá Sai (Mispricing)")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Xóa Cache & Tải Lại Dữ Liệu"):
    st.cache_data.clear()
    st.sidebar.success("Đã xóa bộ nhớ tạm! Vui lòng thao tác lại.")

st.sidebar.markdown("---")
menu = st.sidebar.radio("Chọn chức năng:", ["A. Phân tích Cổ phiếu Riêng lẻ", "B. Cảnh báo Sập gãy (Rổ VN30)", "C. Cảnh báo Định giá sai Rủi ro (Toàn thị trường)"])

end_date = datetime.datetime.now()
plot_start_date = st.sidebar.date_input("Ngày bắt đầu biểu đồ:", pd.to_datetime("2019-01-01"))
data_start_date = pd.to_datetime(plot_start_date) - datetime.timedelta(days=(365 * WINDOW_YEARS) + 30)

rangeselector_dict = dict(
    buttons=list([
        dict(count=30, label="30 Ngày", step="day", stepmode="backward"),
        dict(count=60, label="60 Ngày", step="day", stepmode="backward"),
        dict(count=1, label="1 Năm", step="year", stepmode="backward"),
        dict(count=3, label="3 Năm", step="year", stepmode="backward"),
        dict(step="all", label="Tất cả")
    ]), bgcolor="#e5e7eb", activecolor="#9ca3af"
)

# ------------------------------------------------------------------------------
# MỤC A: CỔ PHIẾU RIÊNG LẺ
# ------------------------------------------------------------------------------
if menu == "A. Phân tích Cổ phiếu Riêng lẻ":
    ticker = st.text_input("Nhập mã cổ phiếu (Kèm .VN):", "MBB.VN").upper()
    if st.button("Chạy Phân Tích"):
        with st.spinner(f"Đang xử lý dữ liệu cho {ticker}..."):
            df_price = fetch_data([ticker], data_start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            if df_price.empty: st.error("Lỗi dữ liệu Yahoo Finance.")
            else:
                actual_start = df_price.index[0].strftime("%Y-%m-%d")
                st.info(f"Dữ liệu thực tế tải được từ hệ thống bắt đầu từ ngày: **{actual_start}**")
                
                df_return, df_var, df_es, _ = calculate_risk_metrics(df_price)
                plot_mask = df_return.index >= pd.to_datetime(plot_start_date)
                
                p_ret = df_return[plot_mask][ticker]
                p_std20 = p_ret.rolling(window=20, min_periods=1).std() * -1
                
                p_var = df_var[plot_mask][ticker]
                p_es = df_es[plot_mask][ticker]
                
                fig_ply = go.Figure()
                fig_ply.add_trace(go.Scatter(x=p_std20.index, y=p_std20, mode='lines', name='-20d Stdev', line=dict(color='gray', width=1.5, dash='dot')))
                fig_ply.add_trace(go.Scatter(x=p_var.index, y=p_var, mode='lines', name='CF VaR 95%', line=dict(color='red', dash='dash')))
                fig_ply.add_trace(go.Scatter(x=p_es.index, y=p_es, mode='lines', name='Robust ES', line=dict(color='purple'), fill='tonexty', fillcolor='rgba(128, 0, 128, 0.15)'))
                fig_ply.update_layout(title=f'Băng thông Rủi ro (Risk Band) {ticker}', template='plotly_white', hovermode='x unified', plot_bgcolor='white', paper_bgcolor='white', font=dict(color='black'))
                fig_ply.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', rangeselector=rangeselector_dict)
                fig_ply.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                st.plotly_chart(fig_ply, use_container_width=True, theme=None)
                
                fig_mpl, ax = plt.subplots(figsize=(12, 6))
                ax.plot(p_std20.index, p_std20, color='gray', linestyle=':', linewidth=1.5, label='-20d Stdev')
                ax.plot(p_var.index, p_var, color='red', linestyle='--', label='CF VaR 95%')
                ax.plot(p_es.index, p_es, color='purple', label='Robust ES')
                ax.fill_between(p_var.index, p_var, p_es, color='purple', alpha=0.1)
                ax.set_title(f'Băng thông Cực Trị (Extreme Band) {ticker}', fontweight='bold')
                ax.legend(loc='upper left')
                ax.grid(alpha=0.3)
                
                pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                with PdfPages(pdf_file.name) as pdf:
                    pdf.savefig(fig_mpl, bbox_inches='tight')
                with open(pdf_file.name, "rb") as file:
                    st.download_button("Tải Báo Cáo PDF", data=file, file_name=f"{ticker}_Risk_Report.pdf", mime="application/pdf")

# ------------------------------------------------------------------------------
# MỤC B: VN30 - BÁO ĐỘNG SẬP GÃY TRỰC TIẾP
# ------------------------------------------------------------------------------
elif menu == "B. Cảnh báo Sập gãy (Rổ VN30)":
    if st.button("Quét Rủi Ro Hệ Thống"):
        with st.spinner("Đang tải dữ liệu Rổ VN30..."):
            df_price = fetch_data(VN30_TICKERS, data_start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            if not df_price.empty:
                df_return, df_var, df_es, df_spread = calculate_risk_metrics(df_price)
                plot_mask = df_return.index >= pd.to_datetime(plot_start_date)
                
                df_breach = df_return[plot_mask] < df_var[plot_mask]
                stress_index = (df_breach.sum(axis=1) / len(VN30_TICKERS)) * 100
                
                st.subheader("Chỉ Số Lây Lan Khủng Hoảng (Crash / Systemic Stress)")
                fig_ply = go.Figure()
                fig_ply.add_trace(go.Scatter(x=stress_index.index, y=stress_index, mode='lines', name='% Cổ phiếu thủng VaR', line=dict(color='teal', width=2)))
                fig_ply.add_hline(y=STRESS_THRESHOLD_VN30 * 100, line_dash="dash", line_color="red", annotation_text="Ngưỡng Báo động Đỏ (40%)")
                fig_ply.update_layout(template='plotly_white', yaxis=dict(range=[0, 105]), hovermode='x unified', plot_bgcolor='white', paper_bgcolor='white', font=dict(color='black'))
                fig_ply.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', rangeselector=rangeselector_dict)
                fig_ply.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                st.plotly_chart(fig_ply, use_container_width=True, theme=None)
                
                latest_ret, latest_var, latest_es = df_return.iloc[-1], df_var.iloc[-1], df_es.iloc[-1]
                risk_table = pd.DataFrame({
                    'Return (%)': latest_ret * 100, 'CF VaR 95% (%)': latest_var * 100, 'ES (%)': latest_es * 100,
                    'Tình trạng': np.where(latest_ret < latest_var, 'Cảnh báo Lây lan', 'Bình thường')
                }).round(2).dropna().sort_values(by=['Tình trạng', 'Return (%)'])
                
                # Hàm in đậm, bỏ tô màu
                def highlight_crash(row):
                    if row['Tình trạng'] == 'Cảnh báo Lây lan': return ['font-weight: bold'] * len(row)
                    return [''] * len(row)
                
                st.dataframe(risk_table.style.apply(highlight_crash, axis=1), use_container_width=True)
                
                fig1_mpl, ax1 = plt.subplots(figsize=(14, 5))
                ax1.plot(stress_index.index, stress_index, color='teal', linewidth=1.5, label='% Cổ phiếu thủng VaR')
                ax1.axhline(y=STRESS_THRESHOLD_VN30 * 100, color='red', linestyle='--', label=f'Ngưỡng Hệ thống ({STRESS_THRESHOLD_VN30*100}%)')
                ax1.fill_between(stress_index.index, stress_index, STRESS_THRESHOLD_VN30 * 100, where=(stress_index >= STRESS_THRESHOLD_VN30 * 100), color='red', alpha=0.3)
                ax1.set_title("Sự lây lan diện rộng - Rổ VN30", fontweight='bold')
                ax1.set_ylabel("% Số lượng cổ phiếu")
                ax1.legend()
                ax1.grid(alpha=0.3)
                
                pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                with PdfPages(pdf_file.name) as pdf:
                    pdf.savefig(fig1_mpl, bbox_inches='tight')
                    
                    fig2, ax2 = plt.subplots(figsize=(12, 10))
                    ax2.axis('off')
                    ax2.set_title(f"Báo cáo Bảng Rủi ro - {end_date.strftime('%Y-%m-%d')}", fontweight='bold', pad=20)
                    
                    table_data = risk_table.head(30) 
                    cell_text = [table_data.iloc[row].astype(str).tolist() for row in range(len(table_data))]
                    table = ax2.table(cellText=cell_text, colLabels=table_data.columns, rowLabels=table_data.index, loc='center', cellLoc='center')
                    table.auto_set_font_size(False)
                    table.set_fontsize(10)
                    table.scale(1.0, 1.5) 
                    
                    # In đậm trong file PDF
                    for (row, col), cell in table.get_celld().items():
                        if row > 0: 
                            if table_data.iloc[row-1]['Tình trạng'] == 'Cảnh báo Lây lan':
                                cell.get_text().set_fontweight('bold')
                                
                    pdf.savefig(fig2, bbox_inches='tight') 
                    
                with open(pdf_file.name, "rb") as file:
                    st.download_button("Tải Báo Cáo Rủi Ro (PDF)", data=file, file_name="Systemic_Risk_Report.pdf", mime="application/pdf")

# ------------------------------------------------------------------------------
# MỤC C: TOÀN THỊ TRƯỜNG - CHỈ BÁO ĐỊNH GIÁ SAI RỦI RO (LỌC PHA TÍCH LŨY)
# ------------------------------------------------------------------------------
elif menu == "C. Cảnh báo Định giá sai Rủi ro (Toàn thị trường)":
    if st.button("Quét Định Giá Rủi Ro"):
        with st.spinner("Đang tính toán ma trận định giá rủi ro..."):
            df_price = fetch_data(ALL_MARKET_TICKERS, data_start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            if not df_price.empty:
                df_return, df_var, df_es, df_spread = calculate_risk_metrics(df_price)
                
                # BỘ LỌC ĐỘNG LƯỢNG (TREND FILTER): Dùng MA 6 tháng (126 phiên)
                # Chỉ những mã đang giao dịch cao hơn mức trung bình 6 tháng mới bị xếp vào diện "Bong bóng/Chủ quan"
                df_ma126 = df_price.rolling(window=126, min_periods=63).mean()
                
                plot_mask = df_spread.index >= pd.to_datetime(plot_start_date)
                df_spread_plot = df_spread[plot_mask]
                df_price_plot = df_price[plot_mask]
                df_ma126_plot = df_ma126[plot_mask]
                
                bank_tickers = [t for t in MARKET_TICKERS['Ngân Hàng'] if t in df_spread_plot.columns]
                non_bank_tickers = [t for t in df_spread_plot.columns if t not in bank_tickers]
                
                # Lấy Spread rủi ro cao nhất của nhóm Ngân Hàng làm chuẩn
                bank_spread_benchmark = df_spread_plot[bank_tickers].max(axis=1)
                
                # Điều kiện 1: Nén rủi ro (Spread <= Bank_Max * 1.25)
                is_spread_compressed = df_spread_plot[non_bank_tickers].lt(bank_spread_benchmark * 1.25, axis=0)
                
                # Điều kiện 2: Thị trường đẩy giá (Price > MA126) - Lọc bỏ pha tích lũy/chán nản
                is_uptrend = df_price_plot[non_bank_tickers] > df_ma126_plot[non_bank_tickers]
                
                # Cổ phiếu bị định giá sai (Mispriced) khi thỏa mãn CẢ 2 điều kiện
                mispriced_matrix = is_spread_compressed & is_uptrend
                
                complacency_index = (mispriced_matrix.sum(axis=1) / len(non_bank_tickers)) * 100
                
                st.subheader("Chỉ Số Ngủ Quên / Định Giá Sai Rủi Ro (Complacency Index)")
                st.markdown("""
                *Chỉ số này đã tích hợp **Bộ lọc Động lượng (Trend Filter)**. Nó loại bỏ pha Tích lũy (nơi rủi ro thấp tự nhiên do thanh khoản cạn) và chỉ báo động khi nhà đầu tư đẩy giá cổ phiếu lên cao (Uptrend) nhưng lại quên định giá rủi ro (Spread bị nén ngang Ngân hàng).*
                """)
                fig_ply = go.Figure()
                fig_ply.add_trace(go.Scatter(x=complacency_index.index, y=complacency_index, mode='lines', name='% Cổ phiếu Mispriced', line=dict(color='darkorange', width=2)))
                fig_ply.add_hline(y=COMPLACENCY_THRESHOLD_MKT * 100, line_dash="dash", line_color="red", annotation_text="Ngưỡng Nguy hiểm (50%)")
                fig_ply.update_layout(template='plotly_white', yaxis=dict(range=[0, 105]), hovermode='x unified', plot_bgcolor='white', paper_bgcolor='white', font=dict(color='black'))
                fig_ply.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', rangeselector=rangeselector_dict)
                fig_ply.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                st.plotly_chart(fig_ply, use_container_width=True, theme=None)
                
                # Bảng rủi ro hiện tại
                latest_spread = df_spread.iloc[-1]
                latest_bank_spread_max = bank_spread_benchmark.iloc[-1]
                latest_price = df_price.iloc[-1]
                latest_ma126 = df_ma126.iloc[-1]
                
                risk_data = []
                for ticker in ALL_MARKET_TICKERS:
                    if ticker not in latest_spread.index or pd.isna(latest_spread[ticker]): continue
                    sector = [s for s, t_list in MARKET_TICKERS.items() if ticker in t_list][0]
                    sprd = latest_spread[ticker] * 100
                    
                    if ticker in bank_tickers: 
                        status = "Benchmark (Bank)"
                    else: 
                        cond_spread = latest_spread[ticker] <= (latest_bank_spread_max * 1.1)
                        cond_uptrend = latest_price[ticker] > latest_ma126[ticker]
                        
                        if cond_spread and cond_uptrend: status = "Risk Mispriced"
                        else: status = "Bình thường / Tích lũy"
                        
                    risk_data.append({'Mã': ticker, 'Ngành': sector, 'Spread (%)': round(sprd, 2), 'Tình trạng': status})
                
                risk_table = pd.DataFrame(risk_data).set_index('Mã')
                risk_table = risk_table.sort_values(by=['Tình trạng', 'Ngành', 'Spread (%)'])
                
                # Chỉ in đậm, không tô màu
                def highlight_mispriced(row):
                    if row['Tình trạng'] == 'Risk Mispriced': return ['font-weight: bold'] * len(row)
                    return [''] * len(row)
                
                st.subheader(f"Bảng Trạng Thái Định Giá (Max Bank Spread: {latest_bank_spread_max*100:.2f}%)")
                st.dataframe(risk_table.style.apply(highlight_mispriced, axis=1), use_container_width=True)
                
                # Matplotlib PDF
                fig1_mpl, ax1 = plt.subplots(figsize=(14, 5))
                ax1.plot(complacency_index.index, complacency_index, color='darkorange', linewidth=1.5, label='% Cổ phiếu Mispriced')
                ax1.axhline(y=COMPLACENCY_THRESHOLD_MKT * 100, color='red', linestyle='--', label=f'Ngưỡng Nguy hiểm ({COMPLACENCY_THRESHOLD_MKT*100}%)')
                ax1.fill_between(complacency_index.index, complacency_index, COMPLACENCY_THRESHOLD_MKT * 100, where=(complacency_index >= COMPLACENCY_THRESHOLD_MKT * 100), color='red', alpha=0.3)
                ax1.set_title("Chỉ Số Ngủ Quên (Complacency Index) - Toàn thị trường", fontweight='bold')
                ax1.set_ylabel("% Số lượng cổ phiếu")
                ax1.legend()
                ax1.grid(alpha=0.3)
                
                pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                with PdfPages(pdf_file.name) as pdf:
                    pdf.savefig(fig1_mpl, bbox_inches='tight')
                    
                    fig2, ax2 = plt.subplots(figsize=(12, 10))
                    ax2.axis('off')
                    ax2.set_title(f"Bảng Trạng Thái Định Giá - {end_date.strftime('%Y-%m-%d')}", fontweight='bold', pad=20)
                    
                    table_data = risk_table.head(30) 
                    cell_text = [table_data.iloc[row].astype(str).tolist() for row in range(len(table_data))]
                    table = ax2.table(cellText=cell_text, colLabels=table_data.columns, rowLabels=table_data.index, loc='center', cellLoc='center')
                    table.auto_set_font_size(False)
                    table.set_fontsize(10)
                    table.scale(1.0, 1.5) 
                    
                    # Quét và in đậm trong PDF
                    for (row, col), cell in table.get_celld().items():
                        if row > 0: 
                            if table_data.iloc[row-1]['Tình trạng'] == 'Risk Mispriced':
                                cell.get_text().set_fontweight('bold')
                                
                    pdf.savefig(fig2, bbox_inches='tight') 
                    
                with open(pdf_file.name, "rb") as file:

                    st.download_button("Tải Báo Cáo Định Giá Rủi (PDF)", data=file, file_name="Mispricing_Risk_Report.pdf", mime="application/pdf")



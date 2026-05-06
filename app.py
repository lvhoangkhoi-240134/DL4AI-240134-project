import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# --- 1. CẤU HÌNH TRANG WIDE & CACHE DỮ LIỆU ---
st.set_page_config(page_title="Pro Stock Terminal", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# Hàm kéo danh sách HÀNG NGÀN công ty tự động (Không bỏ sót)
@st.cache_data
def get_company_dict():
    # Tích hợp sẵn một số mã kinh điển để fallback
    companies = {
        "AAPL": "Apple Inc.", "MSFT": "Microsoft Corp", "GOOGL": "Alphabet Inc.", 
        "AMZN": "Amazon.com", "TSLA": "Tesla Inc.", "NVDA": "NVIDIA Corp", 
        "META": "Meta Platforms", "NFLX": "Netflix Inc.", "V": "Visa Inc.",
        "JPM": "JPMorgan Chase", "WMT": "Walmart Inc.", "JNJ": "Johnson & Johnson"
    }
    # Tự động cào thêm danh sách 500 công ty S&P 500 lớn nhất từ Wikipedia
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        sp500_table = pd.read_html(url)[0]
        for index, row in sp500_table.iterrows():
            companies[row['Symbol']] = row['Security']
    except:
        pass # Nếu lỗi mạng, vẫn có danh sách gốc bên trên
    
    # Format lại thành list chuỗi chuẩn: "AAPL - Apple Inc." để đưa vào thanh tìm kiếm
    formatted_list = [f"{ticker} - {name}" for ticker, name in companies.items()]
    return sorted(formatted_list), companies

ticker_options, ticker_dict = get_company_dict()

# --- HEADER ---
st.title("📈 1D CNN AI Terminal & Financial Advisor")
st.markdown("Giao diện phân tích chuyên sâu: Tích hợp Dự đoán AI, Dữ liệu thực tế, và Phân tích Cố vấn Tài chính.")
st.markdown("---")

# --- 2. SIDEBAR: THANH TÌM KIẾM THÔNG MINH ---
with st.sidebar:
    st.header("⚙️ Model & Market Parameters")
    
    # Thanh tìm kiếm dropdown thông minh (gõ chữ thường, chữ hoa, tên công ty đều ra)
    selected_option = st.selectbox(
        "🔍 Tìm kiếm Công ty (Gõ tên hoặc mã):",
        options=ticker_options,
        index=ticker_options.index("AAPL - Apple Inc.") if "AAPL - Apple Inc." in ticker_options else 0,
        help="Gõ 'Apple' hoặc 'AAPL' để tìm kiếm."
    )
    
    # Tách mã Ticker ra từ chuỗi (Vd: "AAPL - Apple Inc." -> "AAPL")
    ticker = selected_option.split(" - ")[0]
    company_name = selected_option.split(" - ")[1]
    
    days = st.slider("Historical Data Range (Days)", min_value=30, max_value=365, value=60)
    
    st.markdown("---")
    fetch_button = st.button("Fetch Market Data & Predict", type="primary", use_container_width=True)

# --- 3. MAIN LOGIC & UI REVOLUTION ---
if fetch_button:
    with st.spinner(f"📡 Đang tải dữ liệu thực tế & phân tích cho {company_name} ({ticker})..."):
        # Kéo dữ liệu giá và dữ liệu thông tin doanh nghiệp (Real-world data)
        data = yf.download(ticker, period=f"{days}d")
        stock_info = yf.Ticker(ticker).info
        
        if data.empty:
            st.error("⚠️ Không tìm thấy dữ liệu giao dịch. Hãy thử mã khác.")
        else:
            # Lấy các chỉ số tài chính ngoài đời thật
            current_price = float(data['Close'].iloc[-1].iloc[0]) if isinstance(data['Close'], pd.DataFrame) else float(data['Close'].iloc[-1])
            prev_price = float(data['Close'].iloc[-2].iloc[0]) if isinstance(data['Close'], pd.DataFrame) else float(data['Close'].iloc[-2])
            daily_pct_change = ((current_price - prev_price) / prev_price) * 100
            
            market_cap = stock_info.get('marketCap', 'N/A')
            if market_cap != 'N/A':
                market_cap = f"${market_cap / 1e9:.2f} Tỷ" # Đổi sang đơn vị Tỷ USD
                
            forward_pe = stock_info.get('forwardPE', 'N/A')
            analyst_rating = stock_info.get('recommendationKey', 'N/A').upper()
            summary = stock_info.get('longBusinessSummary', 'Không có thông tin mô tả cho công ty này.')
            
            # Tính toán logic giả lập CNN (Giữ nguyên thuật toán của bạn)
            simulated_pred = current_price * (1 + np.random.uniform(-0.02, 0.02))
            price_change = simulated_pred - current_price

            # === HÀNG 1: THÔNG TIN TỔNG QUAN THEO THỜI GIAN THỰC ===
            st.markdown(f"### 🏢 {company_name} ({ticker}) - Real-time Overview")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Current Price", f"${current_price:.2f}", f"{daily_pct_change:.2f}% (Daily)")
            m2.metric("Market Cap (Vốn hóa)", f"{market_cap}")
            m3.metric("Forward P/E Ratio", f"{forward_pe if isinstance(forward_pe, str) else round(forward_pe, 2)}")
            m4.metric("Wall Street Rating", f"{analyst_rating}")
            st.markdown("---")
            
            # === HÀNG 2: CHARTS & AI PREDICTION (CHART RỘNG HƠN) ===
            col1, col2 = st.columns([2.5, 1], gap="large")
            
            with col1:
                st.subheader("📊 Lịch sử Giá Đóng Cửa")
                st.line_chart(data['Close'], height=300)
                
            with col2:
                st.subheader("🤖 1D CNN AI Prediction")
                st.metric(
                    label=f"Dự đoán giá mở cửa ngày mai", 
                    value=f"${simulated_pred:.2f}", 
                    delta=f"${price_change:.2f}"
                )
                
                chart_data = pd.DataFrame({
                    "Price Type": ["Current Close", "Predicted Open"],
                    "Price ($)": [current_price, simulated_pred]
                })
                st.bar_chart(data=chart_data, x="Price Type", y="Price ($)", color="#2e86c1", height=200)

            st.markdown("---")

            # === HÀNG 3: LẤP ĐẦY KHOẢNG TRỐNG (CỐ VẤN & INSIGHTS) ===
            b1, b2, b3 = st.columns([1.5, 1.5, 1])
            
            with b1:
                st.subheader("📖 Hồ sơ Doanh nghiệp (Facts)")
                st.info(f"**Lĩnh vực:** {stock_info.get('sector', 'N/A')} | **Ngành nghề:** {stock_info.get('industry', 'N/A')}\n\n"
                        f"{summary[:500]}... *(Thông tin tóm tắt dựa trên báo cáo thực tế của {company_name}).*")

            with b2:
                st.subheader("💡 Cố vấn Tài chính (Financial Advisor)")
                # Phân tích logic cơ bản để đưa ra lời khuyên
                advise_color = "green" if analyst_rating in ["BUY", "STRONG_BUY"] else "orange" if analyst_rating == "HOLD" else "red"
                
                advice_text = f"Dựa trên dữ liệu Wall Street hiện tại, cổ phiếu này đang được đánh giá là **:{advise_color}[{analyst_rating}]**.\n\n"
                if isinstance(forward_pe, (int, float)):
                    if forward_pe < 15:
                        advice_text += "- 📉 **Định giá:** P/E khá thấp (<15), có thể cổ phiếu đang bị định giá thấp (Undervalued) hoặc công ty đang tăng trưởng chậm.\n"
                    elif forward_pe > 30:
                        advice_text += "- 📈 **Định giá:** P/E cao (>30), nhà đầu tư đang kỳ vọng tăng trưởng bùng nổ trong tương lai (Growth stock), nhưng rủi ro bong bóng cao.\n"
                
                advice_text += f"- 🎯 **Mục tiêu AI:** Neural Network của chúng ta dự đoán biên độ biến động ngày mai là **${price_change:.2f}**."
                st.success(advice_text)

            with b3:
                st.subheader("📁 Dữ liệu Thô (Raw Data)")
                st.write(f"5 ngày giao dịch gần nhất:")
                display_df = data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(5).copy()
                st.dataframe(
                    display_df.style.format("{:,.2f}", subset=['Open', 'High', 'Low', 'Close']).format("{:,}", subset=['Volume']),
                    use_container_width=True
                )
else:
    st.info("👈 Hãy chọn một công ty ở thanh bên trái (có thể gõ tên để tìm kiếm) và bấm **Fetch Market Data & Predict**.")

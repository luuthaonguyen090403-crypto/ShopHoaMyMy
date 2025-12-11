import os
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.http import models
from PIL import Image
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import requests
from io import BytesIO

# Load environment variables (for local development)

GEMINI_API_KEY = "AIzaSyA1oL68N3JPFuYFY1yPxcQWYde-utVsLh0"
# Hide deprecation warnings
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

@st.cache_resource
def init_services():
    """Khởi tạo các dịch vụ backend (AI, Qdrant, Embedding)"""
    # Gemini AI
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash')

    # Qdrant client
    from qdrant_client import QdrantClient


    url = "https://295efda4-568d-454f-88ce-962031f3b025.us-east4-0.gcp.cloud.qdrant.io:6333"
    key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0._M0TJiVNyo6GV0LksoqPLWKPznKP_hhsiywPFcq7hCc"
    client = QdrantClient(url=url, api_key=key)

    # Embedding model
    embedding_model = SentenceTransformer("Alibaba-NLP/gte-multilingual-base",
                                          trust_remote_code=True)

    return model, client, embedding_model


def get_vector(text, embedding_model):
    """Sinh embedding cho đoạn text"""
    if not text.strip():
        return []
    embedding = embedding_model.encode(text)
    return embedding.tolist()


def search_flowers(query, client, embedding_model, limit=5):
    """Tìm kiếm hoa trong Qdrant"""
    collection_name = "RAG_HVNH"  # Tên collection cố định
    try:
        query_vector = get_vector(query, embedding_model)
        if not query_vector:
            return []

        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True
        )
        return search_result
    except Exception as e:
        st.error(f"Lỗi tìm kiếm: {e}")
        return []


def format_flower_info(search_results):
    """Định dạng thông tin hoa cho prompt"""
    if not search_results:
        return "Không tìm thấy sản phẩm phù hợp."

    formatted_info = "🌸 **CÁC SẢN PHẨM HOA PHÙ HỢP:**\n\n"

    for i, record in enumerate(search_results, 1):
        payload = record.payload
        score = record.score

        formatted_info += f"**{i}. {payload.get('title', 'Không có tên')}**\n"
        formatted_info += f"   - Giá: {payload.get('price', 'Chưa có giá')}\n"
        formatted_info += f"   - Link: {payload.get('url', 'Không có link')}\n"
        formatted_info += f"   - khuyến mãi: {payload.get('khuyen_mai', 'không có khuyến mãi')}"
        formatted_info += f"   - Độ phù hợp: {score:.2f}\n"
        if payload.get('description'):
            formatted_info += f"   - Mô tả: {payload.get('description')}\n"
        formatted_info += "\n"

    return formatted_info


def create_chatbot_prompt(user_message, flower_info):
    """Tạo prompt chi tiết cho chatbot"""
    prompt = f"""
Bạn là một chuyên gia tư vấn bán hoa tươi tại cửa hàng Hoa Tươi My My. Hãy trả lời câu hỏi của khách hàng một cách nhiệt tình, chuyên nghiệp và hữu ích. 

**THÔNG TIN SẢN PHẨM TÌM ĐƯỢC:**
{flower_info}

**CÂU HỎI CỦA KHÁCH HÀNG:**
{user_message}

**HƯỚNG DẪN TRẢ LỜI:**
1. Chào hỏi thân thiện và cảm ơn khách hàng
2. Tư vấn sản phẩm phù hợp dựa trên thông tin tìm được
3. Giải thích lý do tại sao sản phẩm phù hợp
4. Cung cấp thông tin hình ảnh, đường dẫn giá cả, đặc điểm nổi bật
5. Gợi ý thêm các dịch vụ khác (giao hàng, thiết kế theo yêu cầu)
6. Hỏi thêm về nhu cầu cụ thể để tư vấn tốt hơn
7. Khuyến khích khách hàng liên hệ hoặc đặt hàng

**PHONG CÁCH TRẢ LỜI:**
- Thân thiện, nhiệt tình
- Chuyên nghiệp nhưng gần gũi
- Sử dụng emoji phù hợp
- Tập trung vào nhu cầu khách hàng
- Không spam thông tin

Hãy trả lời bằng tiếng Việt một cách tự nhiên và hữu ích nhất!
"""
    return prompt


# Cấu hình trang
st.set_page_config(
    page_title="🌸 Chatbot Tư Vấn Hoa Tươi",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)


def display_flower_cards(search_results):
    """Hiển thị các sản phẩm hoa dạng card"""
    if not search_results:
        return

    st.subheader("🌸 Sản phẩm gợi ý cho bạn:")

    cols = st.columns(min(len(search_results), 3))

    for i, record in enumerate(search_results):
        with cols[i % 3]:
            payload = record.payload

            # Hiển thị ảnh với size nhỏ và đồng nhất
            if payload.get('image'):
                try:
                    response = requests.get(payload['image'])
                    img = Image.open(BytesIO(response.content))
                    # Resize ảnh về kích thước cố định để đồng nhất
                    img = img.resize((200, 200), Image.Resampling.LANCZOS)
                    st.image(img, width=200)
                except:
                    st.image("https://via.placeholder.com/200x200?text=No+Image", width=200)
            else:
                st.image("https://via.placeholder.com/200x200?text=No+Image", width=200)

            # Thông tin sản phẩm
            st.markdown(f"**{payload.get('title', 'Không có tên')}**")
            st.markdown(f"**{payload.get('title', 'Không có tên')}**")
            st.markdown(f"💰 **Link:** {payload.get('url', 'Không có link')}")
            st.markdown(f"⭐ **Độ phù hợp:** {record.score:.2f}")

            if payload.get('url'):
                st.markdown(f"🔗 [Xem chi tiết]({payload['url']})")

            st.markdown("---")


def main():
    # Khởi tạo backend
    try:
        model, client, embedding_model = init_services()
    except Exception as e:
        st.error(f"Lỗi khởi tạo dịch vụ: {e}")
        st.error("Vui lòng kiểm tra lại cấu hình API keys và endpoints.")
        return

    # Header
    st.title("🌸 Chatbot Tư Vấn Hoa")
    st.markdown("*Chào mừng bạn đến với cửa hàng hoa tươi! Tôi sẽ giúp bạn tìm những bông hoa đẹp nhất.*")

    # Sidebar
    with st.sidebar:
        # Logo section
        try:
            logo_image = Image.open("logo.png")
            st.image(logo_image, width=200, use_container_width=True)
        except:
            st.markdown("""
                <div style="text-align: center; padding: 10px 0 0 0;">
                    <h2 style="margin: 0; color: #FF69B4;">Chatbot Hoa Tươi</h2>
                    <p style="margin: 5px 0; color: #666;">🌸</p>
                </div>
                """, unsafe_allow_html=True)


        st.markdown("---")

        st.header("⚙️ Cài đặt")

        # Search settings
        search_limit = st.slider("Số sản phẩm tìm kiếm", 1, 10, 5)

        st.markdown("---")

        # Quick suggestions
        st.header("💡 Gợi ý tìm kiếm")
        quick_searches = [
            "có hoa lan không",
            "shop có giao nhanh không",
            "có freeship không",
            "hoa giao gấp",
            "hoa tươi khai trương",
            "hoa chúc mừng",
            "giỏ hoa đẹp",
            "hoa viếng",

        ]

        for search in quick_searches:
            if st.button(f"🔍 {search}"):
                st.session_state.user_input = search

    # Khởi tạo lịch sử chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Welcome message
        welcome_msg = """
        Xin chào! 🌸 Tôi là chatbot tư vấn của cửa hàng Hoa.

        Tôi có thể giúp bạn:
        - 🔍 Tìm kiếm hoa theo dịp (sinh nhật, khai trương, chúc mừng...)
        - 💰 Tư vấn giá cả và chất lượng
        - 🎁 Gợi ý quà tặng phù hợp

        Hãy cho tôi biết bạn đang tìm loại hoa nào nhé!
        """
        st.session_state.messages.append({"role": "assistant", "content": welcome_msg})

    # Hiển thị lịch sử chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if user_input := st.chat_input("Hãy cho tôi biết bạn cần tư vấn gì về hoa tươi..."):
        # Thêm tin nhắn user vào lịch sử
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        # Xử lý input user
        with st.chat_message("assistant"):
            with st.spinner("Đang tìm kiếm sản phẩm phù hợp..."):
                # Tìm kiếm hoa
                search_results = search_flowers(user_input, client, embedding_model, search_limit)

                # Định dạng thông tin hoa
                flower_info = format_flower_info(search_results)

                # Tạo prompt và lấy phản hồi
                prompt = create_chatbot_prompt(user_input, flower_info)

                try:
                    response = model.generate_content(prompt)
                    assistant_response = response.text

                    # Hiển thị phản hồi
                    st.markdown(assistant_response)

                    # Hiển thị card sản phẩm
                    if search_results:
                        st.markdown("---")
                        display_flower_cards(search_results)

                except Exception as e:
                    assistant_response = f"Xin lỗi, tôi gặp lỗi khi xử lý yêu cầu: {e}"
                    st.error(assistant_response)

        # Thêm phản hồi assistant vào lịch sử
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

    # Footer
    st.markdown("---")
    st.markdown("*💝 Cảm ơn bạn đã sử dụng dịch vụ tư vấn Hoa!*")

    # Thông tin liên hệ
    with st.expander("📞 Thông tin liên hệ"):
        st.markdown("""
        **🏪 **
        - ☎️ Hotline: 0979.424.145
        - 🌐 Website: https://hoatuoimymy.com/
        - ⏰ Giờ mở cửa: 7:00 - 22:00 hàng ngày
        """)


if __name__ == "__main__":
    main()

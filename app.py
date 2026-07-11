"""
AI 产品 Demo — 智能客服 + 智能旅游规划 (Streamlit 版)
部署到 Streamlit Cloud
"""

import streamlit as st
import os
from datetime import datetime
from openai import OpenAI

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="AI Demo",
    page_icon="🤖",
    layout="wide",
)

# 隐藏 Streamlit 默认 header/footer/menu
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    .stAppDeployButton {display: none;}
    [data-testid="stToolbar"] {display: none;}
    [data-testid="stDecoration"] {display: none;}
    [data-testid="stStatusWidget"] {display: none;}
    [data-testid="stHeader"] {display: none;}
    .st-emotion-cache-1dp5vir {display: none;}
    .st-emotion-cache-15ecox0 {display: none;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 配置
# ============================================================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", st.secrets.get("DEEPSEEK_API_KEY", ""))
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

client = None
if DEEPSEEK_API_KEY:
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

# ============================================================
# AI 核心
# ============================================================

def call_deepseek(messages, temperature=0.7, max_tokens=2048):
    if client:
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"> ⚠️ API 调用失败: {e}"
    return _fallback(messages)

def _fallback(messages):
    last = messages[-1]["content"] if messages else ""
    sys = messages[0]["content"] if messages else ""
    if "客服" in sys:
        if any(w in last for w in ["退", "坏", "问题"]):
            return "已为您生成退换货工单 #RT20260711，顺丰明天上午上门取件 ✅\n\n> 📋 知识库命中：退换货政策v2.3\n> 🎯 意图：退换申请(94%)"
        elif "订单" in last:
            return "订单 DD20260708001 已到达北京分拣中心，预计明天下午送达 🚚"
        return "您好！我是智选商城客服小智，请问有什么可以帮您？😊"
    return "## 🗺️ 旅行规划\n\n**Day 1**：抵达 → 入住 → 探索周边\n**Day 2**：核心景点 → 美食\n\n| 项目 | 费用 |\n|------|------|\n| 交通 | ¥1,200 |\n| 住宿 | ¥1,800 |\n| 合计 | ¥5,000 |\n\n> （演示模式 — 部署 API Key 后获取个性化规划）"


# ============================================================
# 预设场景
# ============================================================
CS_SCENARIOS = {
    "📦 退换货咨询": "我上周买的蓝牙耳机左耳没声音了，能退吗？",
    "🔍 订单查询": "帮我查一下订单号 DD20260708001 到哪了",
    "😤 投诉处理": "快递员态度太差了！把我包裹扔门口就走了，也不打电话！",
    "💡 产品咨询": "我想买个投影仪，5000以内，主要在家看电影用，有什么推荐？",
}

TRAVEL_SCENARIOS = {
    "🏖️ 三亚四天三晚": ("三亚", 4, 5000, "休闲度假"),
    "🏔️ 云南七日深度游": ("云南（昆明-大理-丽江）", 7, 6000, "自然风光+人文"),
    "🏙️ 上海周末两日游": ("上海", 2, 2000, "都市探索+美食"),
}

# ============================================================
# UI
# ============================================================
st.title("智能客服")
st.caption("点击场景按钮快速体验，或直接输入问题")

tab1, tab2 = st.tabs(["智能客服", "智能旅游规划"])

# ========================
# Tab 1: 智能客服
# ========================
with tab1:
    col1, col2 = st.columns([1, 2])

    with col1:
        scenario = None
        for label, msg in CS_SCENARIOS.items():
            if st.button(label, use_container_width=True, key=f"cs_{label}"):
                scenario = msg

    with col2:
        # 初始化聊天历史
        if "cs_messages" not in st.session_state:
            st.session_state.cs_messages = []

        # 快捷场景触发
        if scenario:
            st.session_state.cs_messages = []
            st.session_state.cs_messages.append({"role": "assistant", "content": "您好！我是智选商城智能客服小智 👋 有什么可以帮您的？"})
            st.session_state.pending_cs = scenario
            st.rerun()

        # 显示历史消息
        for msg in st.session_state.cs_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # 处理 pending 消息
        if "pending_cs" in st.session_state and st.session_state.pending_cs:
            user_msg = st.session_state.pop("pending_cs")
            st.session_state.cs_messages.append({"role": "user", "content": user_msg})
            with st.chat_message("user"):
                st.markdown(user_msg)

            # 构建消息
            msgs = [{
                "role": "system",
                "content": "你是智选商城智能客服「小智」。回复末尾附带AI处理过程：知识库命中、意图识别、情绪分析。退换政策：7天无理由、15天换新、30天保修。"
            }]
            for m in st.session_state.cs_messages:
                msgs.append({"role": "user" if m["role"] == "user" else "assistant", "content": m["content"]})

            with st.chat_message("assistant"):
                with st.spinner("思考中..."):
                    resp = call_deepseek(msgs)
                st.markdown(resp)
            st.session_state.cs_messages.append({"role": "assistant", "content": resp})
            st.rerun()

        # 聊天输入
        if user_input := st.chat_input("输入您的问题...", key="cs_input"):
            st.session_state.cs_messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            msgs = [{
                "role": "system",
                "content": "你是智选商城智能客服「小智」。回复末尾附带AI处理过程：知识库命中、意图识别、情绪分析。退换政策：7天无理由、15天换新、30天保修。"
            }]
            for m in st.session_state.cs_messages:
                msgs.append({"role": "user" if m["role"] == "user" else "assistant", "content": m["content"]})

            with st.chat_message("assistant"):
                with st.spinner("思考中..."):
                    resp = call_deepseek(msgs)
                st.markdown(resp)
            st.session_state.cs_messages.append({"role": "assistant", "content": resp})

        # 清空按钮
        if st.session_state.cs_messages:
            if st.button("🗑️ 清空对话", key="clear_cs"):
                st.session_state.cs_messages = []
                st.rerun()

# ========================
# Tab 2: 智能旅游规划
# ========================
with tab2:
    col1, col2 = st.columns([1, 2])

    with col1:
        destination = st.text_input("📍 目的地", value="三亚")
        days = st.slider("📅 天数", 1, 15, 4)
        budget = st.slider("💰 预算（元/人）", 500, 20000, 5000, 500)
        style = st.selectbox("🎯 旅行风格", [
            "休闲度假", "自然风光", "人文历史", "美食探索",
            "户外冒险", "都市购物", "亲子游", "蜜月浪漫"
        ])
        additional = st.text_area("📝 补充要求（可选）", placeholder="素食、无障碍设施...")

        plan_btn = st.button("🚀 生成旅行规划", type="primary", use_container_width=True)

        st.divider()
        for label in TRAVEL_SCENARIOS:
            if st.button(label, use_container_width=True, key=f"tv_{label}"):
                d, da, b, s = TRAVEL_SCENARIOS[label]
                st.session_state.tv_dest = d
                st.session_state.tv_days = da
                st.session_state.tv_budget = b
                st.session_state.tv_style = s
                st.session_state.tv_trigger = True
                st.rerun()

    with col2:
        # 处理快捷场景触发
        should_plan = plan_btn or st.session_state.pop("tv_generate", False)
        if st.session_state.pop("tv_trigger", False):
            # 读取快捷场景填写的值并标记需要生成
            destination = st.session_state.get("tv_dest", "三亚")
            days = st.session_state.get("tv_days", 4)
            budget = st.session_state.get("tv_budget", 5000)
            style = st.session_state.get("tv_style", "休闲度假")
            additional = st.session_state.get("tv_additional", "")
            st.session_state.tv_generate = True
            st.rerun()

        # 历史结果
        if "travel_result" not in st.session_state:
            st.session_state.travel_result = None

        if plan_btn:
            with st.spinner("✈️ 正在规划中..."):
                prompt = f"""请为我规划一趟旅行：

📍 目的地：{destination}
📅 天数：{days}天
💰 预算：{budget}元/人
🎯 风格：{style}
📝 补充：{additional if additional else '无'}

请按以下 Markdown 格式输出：

## 🗺️ 旅行概览
## 📋 每日行程
## 💰 预算明细（表格）
## 💡 实用贴士
## ⚠️ 注意事项"""

                msgs = [
                    {"role": "system", "content": "你是资深旅行规划师，回复使用 Markdown 格式，详细具体可执行。"},
                    {"role": "user", "content": prompt}
                ]
                result = call_deepseek(msgs, temperature=0.8, max_tokens=3072)
                st.session_state.travel_result = result

        if st.session_state.travel_result:
            st.markdown(st.session_state.travel_result)
        else:
            st.info("👈 填写参数后点击「生成旅行规划」，或点击下方快捷场景一键填充")

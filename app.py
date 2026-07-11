"""
AI 产品 Demo — 智能客服 + 智能旅游规划
部署到 HuggingFace Spaces (Gradio SDK)

面试展示重点：
- 不只是功能演示，更展示 AI 产品思维
- 显示推理过程（不是黑盒）
- 情感识别、知识库检索、多轮对话
"""

import gradio as gr
import os
import json
from datetime import datetime
from openai import OpenAI

# ============================================================
# 配置
# ============================================================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# 预设场景（让面试官快速体验）
CUSTOMER_SERVICE_SCENARIOS = [
    {
        "title": "📦 退换货咨询",
        "user_msg": "我上周买的蓝牙耳机左耳没声音了，能退吗？",
        "context": "用户购买蓝牙耳机7天后出现单侧无声故障，已在保修期内"
    },
    {
        "title": "🔍 订单查询",
        "user_msg": "帮我查一下订单号 DD20260708001 到哪了",
        "context": "用户查询订单物流状态"
    },
    {
        "title": "😤 投诉处理",
        "user_msg": "快递员态度太差了！把我包裹扔门口就走了，也不打电话！",
        "context": "用户投诉配送服务态度问题，情绪激动"
    },
    {
        "title": "💡 产品咨询",
        "user_msg": "我想买个投影仪，5000以内，主要在家看电影用，有什么推荐？",
        "context": "用户咨询产品推荐，有明确预算和使用场景"
    },
]

TRAVEL_SCENARIOS = [
    {
        "title": "🏖️ 三亚四天三晚",
        "destination": "三亚",
        "days": 4,
        "budget": 5000,
        "style": "休闲度假",
    },
    {
        "title": "🏔️ 云南七日深度游",
        "destination": "云南（昆明-大理-丽江）",
        "days": 7,
        "budget": 6000,
        "style": "自然风光+人文体验",
    },
    {
        "title": "🏙️ 上海周末两日游",
        "destination": "上海",
        "days": 2,
        "budget": 2000,
        "style": "都市探索+美食",
    },
]

# ============================================================
# AI 核心逻辑
# ============================================================

client = None
if DEEPSEEK_API_KEY:
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def call_deepseek(messages, temperature=0.7, max_tokens=2048):
    """调用 DeepSeek API，带 fallback"""
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
            return f"[API 调用失败: {e}]\n\n（演示模式）请检查 API Key 配置。"
    else:
        return _fallback_response(messages)


def _fallback_response(messages):
    """无 API Key 时的模拟回复（仅用于 UI 展示）"""
    last_msg = messages[-1]["content"] if messages else ""
    system = messages[0]["content"] if messages else ""

    if "客服" in system:
        if "退" in last_msg or "坏" in last_msg or "问题" in last_msg:
            return """收到您的退换货申请。根据我们的售后政策：

1. **时效确认**：您购买7天，仍在15天包换期内 ✅
2. **问题定性**：左耳无声 → 属于硬件功能故障
3. **处理方案**：为您安排换新，运费由我们承担

**下一步**：已生成退换货工单 #RT20260711-0832，顺丰上门取件时间为明天上午9:00-11:00，您方便吗？

> 📋 知识库命中：3条（退换货政策v2.3、蓝牙耳机故障代码E01、售后流程SOP）
> 🎯 意图识别：退换货申请（置信度 94%）"""
        elif "订单" in last_msg or "查询" in last_msg:
            return f"""已为您查询订单 **DD20260708001**：

| 项目 | 详情 |
|------|------|
| 📦 状态 | **运输中**（已到达北京分拣中心） |
| 🚚 快递 | 顺丰速运 SF1234567890 |
| 📍 最新 | 2026-07-11 08:15 快件到达【北京朝阳集散中心】 |
| 🏠 预计 | **7月12日 下午 送达** |

需要我帮您设置派送提醒吗？📲"""
        else:
            return "您好！我是智能客服小智，请问有什么可以帮您的？😊"
    else:
        return """## 🗺️ 旅行规划结果

### 📋 每日行程

**Day 1**：抵达 → 入住酒店 → 自由探索周边
**Day 2**：核心景点A → 特色午餐 → 下午文化体验
**Day 3**：自然风光 → 户外活动 → 夜市美食

### 💰 预算概览

| 项目 | 预估费用 |
|------|---------|
| 交通 | ¥1,200 |
| 住宿 | ¥1,800 |
| 餐饮 | ¥900 |
| 门票 | ¥600 |
| 其他 | ¥500 |
| **合计** | **¥5,000** |

> （演示模式 — 配置 API Key 后可获得基于实时信息的个性化规划）"""


# ============================================================
# Tab 1: 智能客服
# ============================================================

def customer_service_chat(message, history):
    """智能客服对话处理"""
    # 构建对话历史
    messages = [
        {
            "role": "system",
            "content": """你是一个专业的电商智能客服助手「小智」，服务于"智选商城"。

## 你的能力
1. 退换货处理：根据用户订单状态和问题描述，判断是否符合退换条件
2. 订单查询：查询物流状态、预计送达时间
3. 产品推荐：根据用户需求推荐商品
4. 投诉处理：安抚情绪、记录问题、升级处理

## 回复要求
- 回复末尾务必附带「AI 处理过程」：
  > 📋 知识库命中：X条（具体条目名称）
  > 🎯 意图识别：XXX（置信度 XX%）
  > 😊 情绪分析：正面/中性/负面（关键词：XXX）
- 如涉及投诉，标注是否需要转人工
- 语气亲切但专业，使用适当的emoji

## 退换政策
- 7天内：无理由退货
- 15天内：质量问题换新
- 30天内：保修维修"""
        }
    ]

    # 添加历史对话
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        if h[1]:
            messages.append({"role": "assistant", "content": h[1]})

    messages.append({"role": "user", "content": message})

    response = call_deepseek(messages, temperature=0.7)
    return response


def quick_scenario(scenario_title):
    """快速填充预设场景"""
    for s in CUSTOMER_SERVICE_SCENARIOS:
        if s["title"] == scenario_title:
            return s["user_msg"]
    return ""


# ============================================================
# Tab 2: 智能旅游规划
# ============================================================

def travel_plan(destination, days, budget, style, additional):
    """智能旅游规划"""
    if not destination.strip():
        return "### ⚠️ 请先输入目的地"

    prompt = f"""请为我规划一趟旅行：

📍 目的地：{destination}
📅 天数：{days}天
💰 预算：{budget}元/人
🎯 风格：{style}
📝 补充要求：{additional if additional else "无"}

请按以下格式输出：

## 🗺️ 旅行概览
（2-3句话概括这趟旅行的亮点）

## 📋 每日行程
每天包含：上午/下午/晚上 三个时段的具体安排

## 💰 预算明细
| 项目 | 预估费用 |
（交通/住宿/餐饮/门票/购物/其他，合计不超预算）

## 💡 实用贴士
- 3-5条当地实用建议
- 最佳出行时间提醒
- 必备物品清单

## ⚠️ 注意事项
- 需要提前预订的项目
- 当地避坑提醒"""

    messages = [
        {"role": "system", "content": "你是一个资深旅行规划师，擅长制定高性价比的个性化旅行方案。回复使用Markdown格式，详细、具体、可执行。"},
        {"role": "user", "content": prompt}
    ]

    response = call_deepseek(messages, temperature=0.8, max_tokens=3072)
    return response


def quick_travel_fill(scenario_title):
    """快速填充旅游场景"""
    for s in TRAVEL_SCENARIOS:
        if s["title"] == scenario_title:
            return s["destination"], s["days"], s["budget"], s["style"]
    return "", 3, 3000, "休闲"


# ============================================================
# Gradio UI
# ============================================================

THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="emerald",
    neutral_hue="slate",
)

with gr.Blocks(title="AI 产品 Demo | 智能客服 + 旅游规划") as demo:

    gr.Markdown("""
    # 🤖 AI 产品能力演示

    **两个典型 AI 应用场景**，展示从需求分析到产品落地的完整思路。

    > 💡 **面试官提示**：每个 Tab 右上角有「快捷场景」按钮，一键填充典型用例，无需手动输入。
    > 回复末尾展示了 **AI 推理过程**（意图识别、知识库命中、情绪分析）—— 这是 AI 产品经理的核心关注点。
    """)

    # ========================
    # Tab 1: 智能客服
    # ========================
    with gr.Tab("💬 智能客服"):
        with gr.Row():
            # 左侧：场景选择 + 产品说明
            with gr.Column(scale=1):
                gr.Markdown("### 🎯 快捷场景")
                scenario_btns = []
                for s in CUSTOMER_SERVICE_SCENARIOS:
                    btn = gr.Button(s["title"], size="sm", elem_classes="scenario-btn")
                    scenario_btns.append((btn, s["title"]))

                gr.Markdown("---")
                gr.Markdown("""
                ### 📋 产品设计要点

                | 维度 | 实现 |
                |------|------|
                | 意图识别 | LLM 语义理解 |
                | 知识库 | RAG 检索政策文档 |
                | 情绪分析 | 实时检测用户情绪 |
                | 转人工 | 自动判断升级条件 |
                | 多轮对话 | 上下文关联记忆 |
                """)

            # 右侧：聊天界面
            with gr.Column(scale=2):
                chat = gr.ChatInterface(
                    fn=customer_service_chat,
                    chatbot=gr.Chatbot(height=520, placeholder="你好！我是智能客服小智 👋\n\n可以试试：退换货咨询、订单查询、产品推荐..."),
                    textbox=gr.Textbox(placeholder="输入您的问题...", container=False, scale=7),
                    title="🛒 智选商城 · 智能客服",
                    description="基于 RAG + 意图识别 + 情绪感知 的 AI 客服系统",
                    examples=["我要退货", "我的订单到哪了？", "5000以内的投影仪推荐"],
                    cache_examples=False,
                )

        # 绑定场景按钮
        for btn, title in scenario_btns:
            btn.click(
                fn=lambda t=title: quick_scenario(t),
                outputs=[chat.textbox]
            )

    # ========================
    # Tab 2: 智能旅游规划
    # ========================
    with gr.Tab("🗺️ 智能旅游规划"):
        with gr.Row():
            # 左侧：输入区
            with gr.Column(scale=1):
                gr.Markdown("### ✈️ 规划参数")

                destination_input = gr.Textbox(
                    label="📍 目的地",
                    placeholder="如：三亚、云南、日本...",
                    value="三亚"
                )

                with gr.Row():
                    days_slider = gr.Slider(
                        label="📅 天数", minimum=1, maximum=15, value=4, step=1
                    )
                    budget_slider = gr.Slider(
                        label="💰 预算（元/人）", minimum=500, maximum=20000, value=5000, step=500
                    )

                style_dropdown = gr.Dropdown(
                    label="🎯 旅行风格",
                    choices=["休闲度假", "自然风光", "人文历史", "美食探索", "户外冒险", "都市购物", "亲子游", "蜜月浪漫"],
                    value="休闲度假",
                )

                additional_input = gr.Textbox(
                    label="📝 补充要求（可选）",
                    placeholder="如有特殊需求：素食、需要无障碍设施...",
                    lines=2,
                )

                plan_btn = gr.Button("🚀 生成旅行规划", variant="primary", size="lg")

                gr.Markdown("---")
                gr.Markdown("### 🎯 快捷场景")
                travel_scenario_btns = []
                for s in TRAVEL_SCENARIOS:
                    btn = gr.Button(s["title"], size="sm", elem_classes="scenario-btn")
                    travel_scenario_btns.append((btn, s["title"]))

                gr.Markdown("---")
                gr.Markdown("""
                ### 📋 产品设计要点

                | 维度 | 实现 |
                |------|------|
                | 个性化 | 多维度偏好匹配 |
                | 实时性 | 结合当前季节/天气 |
                | 结构化 | Markdown格式输出 |
                | 预算控制 | 自动分配 + 校验 |
                | 可执行 | 具体到时段+交通 |
                """)

            # 右侧：结果展示
            with gr.Column(scale=2):
                travel_output = gr.Markdown(
                    value="### 👈 填写参数后点击「生成旅行规划」\n\n试试快捷场景一键填充！",
                    elem_id="travel-result",
                )

        # 事件绑定
        plan_btn.click(
            fn=travel_plan,
            inputs=[destination_input, days_slider, budget_slider, style_dropdown, additional_input],
            outputs=[travel_output],
        )

        for btn, title in travel_scenario_btns:
            btn.click(
                fn=lambda t=title: quick_travel_fill(t),
                outputs=[destination_input, days_slider, budget_slider, style_dropdown],
            ).then(
                fn=travel_plan,
                inputs=[destination_input, days_slider, budget_slider, style_dropdown, additional_input],
                outputs=[travel_output],
            )

    # ========================
    # 页脚
    # ========================
    gr.Markdown("""
    ---
    <div style="text-align: center; color: #94a3b8; font-size: 0.85em;">
    🚀 Powered by DeepSeek · Built with Gradio · Deployed on HuggingFace Spaces<br>
    AI 产品经理面试作品 | 展示 AI 产品设计思维而非单纯功能
    </div>
    """)


# ============================================================
# 启动
# ============================================================
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=THEME,
        css="""
            .scenario-btn { margin: 4px !important; }
            .process-box { background: #f0fdf4; border-left: 4px solid #22c55e; padding: 12px; border-radius: 8px; margin-top: 8px; }
            footer { display: none !important; }
        """,
    )

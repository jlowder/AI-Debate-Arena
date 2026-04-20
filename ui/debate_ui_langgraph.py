import time

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from config_utils import load_config, save_config
from debate_agents_langgraph import (
    DebateState,
    final_judge_node,
    judge_node,
    opponent_node,
    proponent_node,
    router,
)


def run_debate_graph(
    topic,
    pro_temp=0.8,
    con_temp=0.8,
    judge_temp=0.5,
    max_rounds=3,
    model_name="gemma4:e2b",
    api_token="",
    live=True,
):
    # Create placeholder for live updates
    debate_placeholder = st.empty() if live else None

    # Initialize conversation history for display
    current_content = f"<div style='background-color: #f0f0f0; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>Moderator:</strong><br>The topic is: <strong>{topic}</strong> Start the debate.</div>\n\n"
    if live:
        debate_placeholder.markdown(current_content, unsafe_allow_html=True)
        time.sleep(1)

    # We need to wrap the nodes to update the UI
    def ui_proponent_node(state):
        res = proponent_node(state)
        if live:
            nonlocal current_content
            # Extract the new message
            new_msg = res["messages"][-1].content
            display_msg = new_msg.replace("Proponent: ", "")

            current_content += f"<h3>Round {res['round_count']}</h3>\n"
            current_content += f"<div style='background-color: #e8f5e8; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>Proponent:</strong><br>{display_msg}</div>\n\n"
            debate_placeholder.markdown(current_content, unsafe_allow_html=True)
            time.sleep(1)
        return res

    def ui_opponent_node(state):
        res = opponent_node(state)
        if live:
            nonlocal current_content
            new_msg = res["messages"][-1].content
            display_msg = new_msg.replace("Opponent: ", "")

            current_content += f"<div style='background-color: #fce8e8; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>Opponent:</strong><br>{display_msg}</div>\n\n"
            debate_placeholder.markdown(current_content, unsafe_allow_html=True)
            time.sleep(1)
        return res

    def ui_judge_node(state):
        if live:
            # Before calling judge, show "Checking..."
            nonlocal current_content
            if state["round_count"] < state["max_rounds"]:
                check_content = (
                    current_content
                    + f"<div style='background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>Judge (Interim Decision):</strong><br>Checking if debate should continue...</div>\n\n"
                )
                debate_placeholder.markdown(check_content, unsafe_allow_html=True)
                time.sleep(1)

        res = judge_node(state)

        if live:
            if state["round_count"] < state["max_rounds"]:
                judge_reason = res.get("judge_reason", "")
                if not res["should_continue"]:
                    current_content += f"<div style='background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>Judge (Interim Decision):</strong><br>🛑 {judge_reason}<br><em>Proceeding to final judgment early.</em></div>\n\n"
                else:
                    current_content += f"<div style='background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>Judge (Interim Decision):</strong><br>🔄 {judge_reason}</div>\n\n"

                current_content += "<hr style='margin: 20px 0;'>\n\n"
                debate_placeholder.markdown(current_content, unsafe_allow_html=True)
                time.sleep(1)

        return res

    def ui_final_judge_node(state):
        if live:
            nonlocal current_content
            current_content += f"<h3>⚖️ Final Judgment</h3>\n"
            current_content += f"<div style='background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>Judge:</strong><br>Generating final judgment...</div>\n\n"
            debate_placeholder.markdown(current_content, unsafe_allow_html=True)
            time.sleep(1)

        res = final_judge_node(state)

        if live:
            # Remove "Generating..."
            current_content = current_content.replace(
                f"<div style='background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>Judge:</strong><br>Generating final judgment...</div>\n\n",
                "",
            )
            current_content += f"<div style='background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>Judge:</strong><br>{res['final_verdict']}</div>\n\n"
            debate_placeholder.markdown(current_content, unsafe_allow_html=True)
        return res

    # Build the graph
    workflow = StateGraph(DebateState)
    workflow.add_node("proponent", ui_proponent_node)
    workflow.add_node("opponent", ui_opponent_node)
    workflow.add_node("judge", ui_judge_node)
    workflow.add_node("final_judge", ui_final_judge_node)

    workflow.add_edge(START, "proponent")
    workflow.add_edge("proponent", "opponent")
    workflow.add_edge("opponent", "judge")
    workflow.add_conditional_edges(
        "judge", router, {"proponent": "proponent", "final_judge": "final_judge"}
    )
    workflow.add_edge("final_judge", END)

    app = workflow.compile()
    print(app.get_graph().draw_mermaid())

    initial_state = {
        "topic": topic,
        "messages": [
            HumanMessage(
                content=f"The topic is: <strong>{topic}</strong> Start the debate."
            )
        ],
        "round_count": 0,
        "max_rounds": max_rounds,
        "total_tokens": 0,
        "pro_temp": pro_temp,
        "con_temp": con_temp,
        "judge_temp": judge_temp,
        "should_continue": True,
        "judge_reason": "",
        "final_verdict": "",
        "model_name": model_name,
        "api_token": api_token,
    }

    # We use a stream to capture all intermediate states if we wanted to,
    # but since we use nonlocal current_content and ui_nodes, invoke is fine for live updates.
    final_state = app.invoke(initial_state)

    results = {
        "conversation_history": final_state["messages"],
        "final_judgment": final_state["final_verdict"],
        "total_tokens": final_state["total_tokens"],
        "rounds": final_state["round_count"],
        "topic": topic,
        "model_name": final_state.get("model_name", "Unknown"),
    }
    st.session_state.debate_results = results

    if live:
        st.divider()
        st.subheader("📊 Stats")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rounds Completed", final_state["round_count"])
        col2.metric("Total Tokens Used", final_state["total_tokens"])
        col3.metric("Model Used", final_state.get("model_name", "Unknown"))

    return results


def main():
    st.set_page_config(page_title="Debate AI (LangGraph)", page_icon="🗣️", layout="wide")

    # Load config
    config = load_config()
    st.title("🗣️ AI Debate Arena (LangGraph Version)")
    st.markdown(
        "Watch two AI agents debate any topic with a judge overseeing the discussion (using LangGraph)!"
    )

    if "debate_results" not in st.session_state:
        st.session_state.debate_results = None
    if "live_updates" not in st.session_state:
        st.session_state.live_updates = True

    with st.sidebar:
        st.header("⚙️ Settings")

        st.subheader("Ollama Configuration")
        model_name = st.text_input(
            "Model Name", value=config.get("model_name", "gemma4:e2b")
        )
        api_token = st.text_input(
            "API Token (optional)", value=config.get("api_token", ""), type="password"
        )

        st.divider()
        st.subheader("Debate Parameters")
        max_rounds = st.slider("Maximum Rounds", 1, 10, 3)
        pro_temp = st.slider("Proponent Creativity", 0.0, 1.0, 0.9)
        con_temp = st.slider("Opponent Creativity", 0.0, 1.0, 0.7)
        judge_temp = st.slider("Judge Strictness", 0.0, 1.0, 0.3)

        st.divider()
        st.checkbox("Show live debate updates", key="live_updates")

    topic = st.text_input(
        "Enter debate topic:", "C++ is better than Python for security."
    )

    if st.button("🚀 Start Debate", type="primary"):
        if topic.strip():
            # Save config
            save_config({"model_name": model_name, "api_token": api_token})

            if st.session_state.live_updates:
                run_debate_graph(
                    topic,
                    pro_temp,
                    con_temp,
                    judge_temp,
                    max_rounds,
                    model_name=model_name,
                    api_token=api_token,
                    live=True,
                )
            else:
                with st.spinner("Debate in progress..."):
                    try:
                        run_debate_graph(
                            topic,
                            pro_temp,
                            con_temp,
                            judge_temp,
                            max_rounds,
                            model_name=model_name,
                            api_token=api_token,
                            live=False,
                        )
                    except Exception as e:
                        st.error(f"Error running debate: {str(e)}")
        else:
            st.warning("Please enter a debate topic")

    # Display results for non-live or after live finished
    if st.session_state.debate_results and (not st.session_state.live_updates):
        results = st.session_state.debate_results
        st.divider()
        st.header("🎭 Debate Results")

        st.subheader("💬 Conversation")
        st.markdown(
            f"<div style='background-color: #f0f0f0; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>Moderator:</strong><br>The topic is: <strong>{results['topic']}</strong> Start the debate.</div>",
            unsafe_allow_html=True,
        )

        for i, msg in enumerate(results["conversation_history"]):
            if isinstance(msg, HumanMessage):
                continue

            if isinstance(msg, AIMessage):
                content = msg.content
                if content.startswith("Proponent:"):
                    st.markdown(
                        f"<div style='background-color: #e8f5e8; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>Proponent:</strong><br>{content.replace('Proponent: ', '')}</div>",
                        unsafe_allow_html=True,
                    )
                elif content.startswith("Opponent:"):
                    st.markdown(
                        f"<div style='background-color: #fce8e8; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>Opponent:</strong><br>{content.replace('Opponent: ', '')}</div>",
                        unsafe_allow_html=True,
                    )

        st.subheader("⚖️ Final Judgment")
        st.markdown(
            f"<div style='background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>Judge:</strong><br>{results['final_judgment']}</div>",
            unsafe_allow_html=True,
        )

        st.subheader("📊 Stats")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rounds Completed", results["rounds"])
        col2.metric("Total Tokens Used", results["total_tokens"])
        col3.metric("Model Used", results.get("model_name", "Unknown"))


if __name__ == "__main__":
    main()

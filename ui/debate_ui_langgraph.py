import time
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from debate_agents_langgraph import DebateState, proponent_node, opponent_node, judge_node, final_judge_node, router
from langgraph.graph import StateGraph, START, END

def run_debate_live(topic, pro_temp=0.8, con_temp=0.8, judge_temp=0.5, max_rounds=3):
    # Create placeholder for live updates
    debate_placeholder = st.empty()

    # Initialize conversation history for display
    current_content = f"<div style='background-color: #f0f0f0; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>Moderator:</strong><br>The topic is: <strong>{topic}</strong> Start the debate.</div>\n\n"
    debate_placeholder.markdown(current_content, unsafe_allow_html=True)
    time.sleep(1)

    # We need to wrap the nodes to update the UI
    def ui_proponent_node(state):
        res = proponent_node(state)
        nonlocal current_content
        # Extract the new message
        new_msg = res["messages"][-1].content
        # In proponent_node, we prepend "Proponent: " so we remove it for UI display if needed,
        # but the original UI also kept it or handled it.
        # Actually proponent_node adds "Proponent: " to the content.
        display_msg = new_msg.replace("Proponent: ", "")

        current_content += f"<h3>Round {res['round_count']}</h3>\n"
        current_content += f"<div style='background-color: #e8f5e8; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>Proponent:</strong><br>{display_msg}</div>\n\n"
        debate_placeholder.markdown(current_content, unsafe_allow_html=True)
        time.sleep(1)
        return res

    def ui_opponent_node(state):
        res = opponent_node(state)
        nonlocal current_content
        new_msg = res["messages"][-1].content
        display_msg = new_msg.replace("Opponent: ", "")

        current_content += f"<div style='background-color: #fce8e8; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>Opponent:</strong><br>{display_msg}</div>\n\n"
        debate_placeholder.markdown(current_content, unsafe_allow_html=True)
        time.sleep(1)
        return res

    def ui_judge_node(state):
        # Before calling judge, show "Checking..."
        nonlocal current_content
        if state["round_count"] < state["max_rounds"]:
            check_content = current_content + f"<div style='background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>Judge (Interim Decision):</strong><br>Checking if debate should continue...</div>\n\n"
            debate_placeholder.markdown(check_content, unsafe_allow_html=True)
            time.sleep(1)

        res = judge_node(state)

        # We don't have the judge's reasoning in the state in the current langgraph implementation
        # unless we modify it. Let's assume we want to keep it simple or modify langgraph implementation.
        # For now, let's just say if it's continuing or not.

        if state["round_count"] < state["max_rounds"]:
            if not res["should_continue"]:
                current_content += f"<div style='background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>Judge (Interim Decision):</strong><br>🛑 Judgment Ready.<br><em>Proceeding to final judgment early.</em></div>\n\n"
            else:
                current_content += f"<div style='background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>Judge (Interim Decision):</strong><br>🔄 Continue debating...</div>\n\n"

            current_content += "<hr style='margin: 20px 0;'>\n\n"
            debate_placeholder.markdown(current_content, unsafe_allow_html=True)
            time.sleep(1)

        return res

    def ui_final_judge_node(state):
        nonlocal current_content
        current_content += f"<h3>⚖️ Final Judgment</h3>\n"
        current_content += f"<div style='background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>Judge:</strong><br>Generating final judgment...</div>\n\n"
        debate_placeholder.markdown(current_content, unsafe_allow_html=True)
        time.sleep(1)

        res = final_judge_node(state)

        # Remove "Generating..."
        current_content = current_content.replace(f"<div style='background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>Judge:</strong><br>Generating final judgment...</div>\n\n", "")
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
    workflow.add_conditional_edges("judge", router, {"proponent": "proponent", "final_judge": "final_judge"})
    workflow.add_edge("final_judge", END)

    app = workflow.compile()

    initial_state = {
        "topic": topic,
        "messages": [HumanMessage(content=f"The topic is: {topic}. Start the debate.")],
        "round_count": 0,
        "max_rounds": max_rounds,
        "total_tokens": 0,
        "pro_temp": pro_temp,
        "con_temp": con_temp,
        "judge_temp": judge_temp,
        "should_continue": True,
        "final_verdict": ""
    }

    final_state = app.invoke(initial_state)

    st.session_state.debate_results = {
        "conversation_history": final_state["messages"],
        "final_judgment": final_state["final_verdict"],
        "total_tokens": final_state["total_tokens"],
        "rounds": final_state["round_count"],
    }

    st.divider()
    st.subheader("📊 Stats")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rounds Completed", final_state["round_count"])
    col2.metric("Total Tokens Used", final_state["total_tokens"])
    col3.metric("Model Used", "my-gemma")

def main():
    st.set_page_config(page_title="Debate AI (LangGraph)", page_icon="🗣️", layout="wide")
    st.title("🗣️ AI Debate Arena (LangGraph Version)")

    if "debate_results" not in st.session_state:
        st.session_state.debate_results = None

    with st.sidebar:
        st.header("⚙️ Settings")
        max_rounds = st.slider("Maximum Rounds", 1, 10, 3)
        pro_temp = st.slider("Proponent Creativity", 0.0, 1.0, 0.9)
        con_temp = st.slider("Opponent Creativity", 0.0, 1.0, 0.7)
        judge_temp = st.slider("Judge Strictness", 0.0, 1.0, 0.3)

    topic = st.text_input("Enter debate topic:", "C++ is better than Python for security.")

    if st.button("🚀 Start Debate", type="primary"):
        if topic.strip():
            run_debate_live(topic, pro_temp, con_temp, judge_temp, max_rounds)
        else:
            st.warning("Please enter a debate topic")

if __name__ == "__main__":
    main()

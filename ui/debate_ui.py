import time

import ollama
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


class DebateAgent:
    def __init__(self, model_name, system_prompt, temperature=0.8):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature

    def respond(self, input_text, history):
        """Generate a response based on the input and conversation history"""
        # Prepare messages for Ollama call
        messages = [{"role": "system", "content": self.system_prompt}]

        # Convert history to Ollama format
        for msg in history:
            if isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                messages.append({"role": "system", "content": msg.content})

        # Add current input
        messages.append({"role": "user", "content": input_text})

        # Make direct call to Ollama to get token information
        response = ollama.chat(
            model=self.model_name,
            messages=messages,
            options={"temperature": self.temperature},
        )

        # Extract token usage if available
        tokens_used = 0
        if "prompt_eval_count" in response and "eval_count" in response:
            tokens_used = response["prompt_eval_count"] + response["eval_count"]
        elif "eval_count" in response:
            tokens_used = response["eval_count"]

        return response["message"]["content"], tokens_used


def should_continue_debate(judge_agent, conversation_history, max_rounds=3):
    """Ask the judge if the debate should continue or if there's enough information to declare a winner"""
    judge_prompt = """
    Based on the debate so far, do you have enough information to make a final judgment, or should the debaters continue?

    You should recommend stopping the debate ("JUDGMENT READY") if:
    - Both sides have presented at least one argument
    - There has been some attempt at rebuttal
    - You can form a reasonable opinion based on what's been said

    You should recommend continuing ("CONTINUE") only if:
    - One side hasn't presented a meaningful argument
    - There's been no impactful rebuttal from either side
    - Critical information is missing that would be necessary for any judgment

    Respond with either:
    "JUDGMENT READY" if you have enough information to make a final decision
    "CONTINUE" if more rounds would be beneficial

    Additionally, provide a brief explanation (1 sentence) of your reasoning after stating your decision.
    For example: "JUDGMENT READY - Both sides have presented arguments and there's enough information to make a reasonable judgment."
    Or: "CONTINUE - The con side hasn't provided a substantive counter-argument yet."

    Do not include any other text in your response.
    """

    judge_response, _ = judge_agent.respond(judge_prompt, conversation_history)

    # Parse the judge's response
    response_upper = judge_response.strip().upper()
    if "JUDGMENT READY" in response_upper:
        return False, judge_response  # Stop the debate
    else:
        return True, judge_response  # Continue the debate


def run_debate(topic, pro_temp=0.8, con_temp=0.8, judge_temp=0.5, max_rounds=3):
    model_name = "my-gemma"

    # Create debate agents with different personas and temperatures
    pro_agent = DebateAgent(
        model_name,
        "You are an optimistic advocate. Support the topic with logic and enthusiasm. Be brief.",
        pro_temp,
    )

    con_agent = DebateAgent(
        model_name,
        "You are a skeptical critic. Find flaws in the opponent's logic and argue against the topic. Be brief.",
        con_temp,
    )

    judge_agent = DebateAgent(
        model_name,
        "You are a neutral judge who can make reasonable judgments on practical matters. You can make a judgment based on the arguments presented even if they aren't completely exhaustive.",
        judge_temp,
    )

    final_judge_agent = DebateAgent(
        model_name,
        "You are a neutral judge. Summarize the key points of both sides and declare a logical winner.",
        judge_temp,
    )

    # Initialize conversation history
    conversation_history = []
    initial_prompt = f"The topic is: {topic}. Start the debate."
    conversation_history.append(HumanMessage(content=initial_prompt))

    # Track total tokens used
    total_tokens = 0
    rounds_run = 0
    judge_decisions = []

    # Run debate rounds dynamically
    round_count = 0
    while round_count < max_rounds:
        round_count += 1
        rounds_run = round_count

        # Proponent speaks
        pro_input = "Present your argument."
        pro_response, pro_tokens = pro_agent.respond(pro_input, conversation_history)
        total_tokens += pro_tokens
        conversation_history.append(AIMessage(content=f"Proponent: {pro_response}"))

        # Opponent responds
        con_input = "Respond to the proponent's argument."
        con_response, con_tokens = con_agent.respond(con_input, conversation_history)
        total_tokens += con_tokens
        conversation_history.append(AIMessage(content=f"Opponent: {con_response}"))

        # Ask the judge if we should continue
        if round_count < max_rounds:
            judge_continue, judge_reason = should_continue_debate(
                judge_agent, conversation_history, max_rounds
            )
            judge_decisions.append(
                {
                    "round": round_count,
                    "decision": "JUDGMENT READY" if not judge_continue else "CONTINUE",
                    "reason": judge_reason,
                }
            )
            if not judge_continue:
                break

    # Judge's final verdict
    judge_input = "Provide your final judgment based on all arguments presented. Summarize the key points of both sides and declare a logical winner."
    judge_response, judge_tokens = final_judge_agent.respond(
        judge_input, conversation_history
    )
    total_tokens += judge_tokens

    return {
        "conversation_history": conversation_history,
        "final_judgment": judge_response,
        "total_tokens": total_tokens,
        "rounds": rounds_run,
        "judge_interim_decisions": judge_decisions,
    }


def run_debate_live(topic, pro_temp=0.8, con_temp=0.8, judge_temp=0.5, max_rounds=3):
    model_name = "my-gemma"

    # Create debate agents with different personas and temperatures
    pro_agent = DebateAgent(
        model_name,
        "You are an optimistic advocate. Support the topic with logic and enthusiasm. Be brief.",
        pro_temp,
    )

    con_agent = DebateAgent(
        model_name,
        "You are a skeptical critic. Find flaws in the opponent's logic and argue against the topic. Be brief.",
        con_temp,
    )

    judge_agent = DebateAgent(
        model_name,
        "You are a decisive judge who can make reasonable judgments on practical matters. You can make a judgment based on the arguments presented even if they aren't completely exhaustive. When asked if the debate should continue, provide a brief explanation of your reasoning.",
        judge_temp,
    )

    final_judge_agent = DebateAgent(
        model_name,
        "You are a neutral judge. Summarize the key points of both sides and declare a logical winner.",
        judge_temp,
    )

    # Create placeholder for live updates
    debate_placeholder = st.empty()

    # Initialize conversation history
    conversation_history = []
    initial_prompt = f"The topic is: {topic}  Start the debate."
    conversation_history.append(HumanMessage(content=initial_prompt))

    # Track total tokens used
    total_tokens = 0
    rounds_run = 0

    # Display initial prompt with chat styling
    current_content = f"<div style='background-color: #f0f0f0; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>Moderator:</strong><br>{initial_prompt}</div>\n\n"
    debate_placeholder.markdown(current_content, unsafe_allow_html=True)
    time.sleep(1)

    # Run debate rounds dynamically
    round_count = 0
    while round_count < max_rounds:
        round_count += 1
        rounds_run = round_count

        # Display round header
        current_content += f"<h3>Round {round_count}</h3>\n"
        debate_placeholder.markdown(current_content, unsafe_allow_html=True)
        time.sleep(0.5)

        # Proponent speaks
        pro_input = "Present your argument."
        pro_response, pro_tokens = pro_agent.respond(pro_input, conversation_history)
        total_tokens += pro_tokens
        conversation_history.append(AIMessage(content=f"Proponent: {pro_response}"))

        # Display proponent response with green background
        current_content += f"<div style='background-color: #e8f5e8; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>Proponent:</strong><br>{pro_response}</div>\n\n"
        debate_placeholder.markdown(current_content, unsafe_allow_html=True)
        time.sleep(1)

        # Opponent responds
        con_input = "Respond to the proponent's argument."
        con_response, con_tokens = con_agent.respond(con_input, conversation_history)
        total_tokens += con_tokens
        conversation_history.append(AIMessage(content=f"Opponent: {con_response}"))

        # Display opponent response with red background
        current_content += f"<div style='background-color: #fce8e8; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>Opponent:</strong><br>{con_response}</div>\n\n"
        debate_placeholder.markdown(current_content, unsafe_allow_html=True)
        time.sleep(1)

        # Ask the judge if we should continue (except on last round)
        if round_count < max_rounds:
            # Display judge checking message with blue background
            current_content += f"<div style='background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>Judge (Interim Decision):</strong><br>Checking if debate should continue...</div>\n\n"
            debate_placeholder.markdown(current_content, unsafe_allow_html=True)
            time.sleep(1)

            judge_continue, judge_reason = should_continue_debate(
                judge_agent, conversation_history, max_rounds
            )

            # Display judge's interim decision with blue background
            if not judge_continue:
                current_content += f"<div style='background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>Judge (Interim Decision):</strong><br>🛑 JUDGMENT READY - {judge_reason}<br><em>Proceeding to final judgment early.</em></div>\n\n"
                debate_placeholder.markdown(current_content, unsafe_allow_html=True)
                time.sleep(1)
                break
            else:
                current_content += f"<div style='background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>Judge (Interim Decision):</strong><br>🔄 CONTINUE - {judge_reason}</div>\n\n"
                debate_placeholder.markdown(current_content, unsafe_allow_html=True)
                time.sleep(1)

        # Add spacing between rounds
        current_content += "<hr style='margin: 20px 0;'>\n\n"
        debate_placeholder.markdown(current_content, unsafe_allow_html=True)

    # Judge's final verdict
    current_content += f"<h3>⚖️ Final Judgment</h3>\n"
    current_content += f"<div style='background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>Judge:</strong><br>Generating final judgment...</div>\n\n"
    debate_placeholder.markdown(current_content, unsafe_allow_html=True)
    time.sleep(1)

    judge_input = "Provide your final judgment based on all arguments presented. Summarize the key points of both sides and declare a logical winner."
    judge_response, judge_tokens = final_judge_agent.respond(
        judge_input, conversation_history
    )
    total_tokens += judge_tokens

    # Display final judgment with blue background
    current_content += f"<div style='background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>Judge:</strong><br>{judge_response}</div>\n\n"
    debate_placeholder.markdown(current_content, unsafe_allow_html=True)

    # Store results for later display
    st.session_state.debate_results = {
        "conversation_history": conversation_history,
        "final_judgment": judge_response,
        "total_tokens": total_tokens,
        "rounds": rounds_run,
    }


def main():

    st.set_page_config(page_title="Debate AI", page_icon="🗣️", layout="wide")

    st.title("🗣️ AI Debate Arena")
    st.markdown(
        "Watch two AI agents debate any topic with a judge overseeing the discussion!"
    )

    # Initialize session state
    if "debate_results" not in st.session_state:
        st.session_state.debate_results = None
    if "live_updates" not in st.session_state:
        st.session_state.live_updates = False

    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        max_rounds = st.slider("Maximum Rounds", 1, 10, 3)
        pro_temp = st.slider("Proponent Creativity", 0.0, 1.0, 0.9)
        con_temp = st.slider("Opponent Creativity", 0.0, 1.0, 0.7)
        judge_temp = st.slider("Judge Strictness", 0.0, 1.0, 0.3)

        st.divider()
        st.checkbox("Show live debate updates", key="live_updates")

        st.divider()
        st.markdown("💡 **Tips:**")
        st.markdown("- Try controversial topics for more engaging debates")
        st.markdown("- Adjust creativity settings to change argument styles")
        st.markdown("- Lower judge strictness for more lenient judging")

    # Main input area
    topic = st.text_input(
        "Enter debate topic:",
        "Should people perform their own surgeries to save costs?",
    )

    if st.button("🚀 Start Debate", type="primary"):
        if topic.strip():
            if st.session_state.live_updates:
                # Run debate with live updates
                run_debate_live(
                    topic,
                    pro_temp=pro_temp,
                    con_temp=con_temp,
                    judge_temp=judge_temp,
                    max_rounds=max_rounds,
                )
            else:
                with st.spinner("Debate in progress..."):
                    try:
                        results = run_debate(
                            topic,
                            pro_temp=pro_temp,
                            con_temp=con_temp,
                            judge_temp=judge_temp,
                            max_rounds=max_rounds,
                        )
                        st.session_state.debate_results = results
                    except Exception as e:
                        st.error(f"Error running debate: {str(e)}")
        else:
            st.warning("Please enter a debate topic")

        # Display results
        if st.session_state.debate_results and not st.session_state.live_updates:
            st.divider()
            st.header("🎭 Debate Results")

            results = st.session_state.debate_results

            # Display conversation
            st.subheader("💬 Conversation")

            # Show initial prompt with chat styling
            st.markdown(
                f"<div style='background-color: #f0f0f0; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>Moderator:</strong><br>The topic is: {topic}. Start the debate.</div>",
                unsafe_allow_html=True,
            )

            # Show each exchange
            for i, msg in enumerate(results["conversation_history"]):
                if isinstance(msg, HumanMessage):
                    continue  # Skip initial prompt as we handled it above

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

                # Add spacing between exchanges
                if i < len(results["conversation_history"]) - 1:
                    st.markdown("---")

            # Show judge interim decisions if any
            if results.get("judge_interim_decisions"):
                st.subheader("👨‍⚖️ Judge Interim Decisions")
                for decision in results["judge_interim_decisions"]:
                    decision_text = f"<div style='background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>Round {decision['round']}:</strong><br>"
                    if decision["decision"] == "JUDGMENT READY":
                        decision_text += (
                            f"🛑 {decision['decision']} - {decision['reason']}"
                        )
                    else:
                        decision_text += (
                            f"🔄 {decision['decision']} - {decision['reason']}"
                        )
                    decision_text += "</div>"
                    st.markdown(decision_text, unsafe_allow_html=True)

            # Show final judgment
            st.subheader("⚖️ Final Judgment")
            st.markdown(
                f"<div style='background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>Judge:</strong><br>{results['final_judgment']}</div>",
                unsafe_allow_html=True,
            )

            # Show stats
            st.subheader("📊 Stats")
            col1, col2, col3 = st.columns(3)
            col1.metric("Rounds Completed", results["rounds"])
            col2.metric("Total Tokens Used", results["total_tokens"])
            col3.metric("Model Used", "my-gemma")


if __name__ == "__main__":
    main()

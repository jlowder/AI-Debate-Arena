import ollama
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

    Consider:
    - Have both sides presented substantial arguments?
    - Has there been adequate rebuttal and counter-rebuttal?
    - Are you confident in your ability to declare a winner based on the current arguments?

    Respond with exactly one of these two phrases:
    "CONTINUE" if you think more rounds would be beneficial
    "JUDGMENT READY" if you have enough information to make a final decision

    Do not include any other text in your response.
    """

    judge_response, _ = judge_agent.respond(judge_prompt, conversation_history)

    # Parse the judge's response
    response_upper = judge_response.strip().upper()
    if "CONTINUE" not in response_upper:
        return False  # Stop the debate
    else:
        return True  # Continue the debate


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
        "You are a neutral judge. You will evaluate if enough information has been presented to make a final decision, or if more debate rounds are needed.",
        judge_temp,
    )

    final_judge_agent = DebateAgent(
        model_name,
        "You are a neutral judge. Summarize the key points of both sides and declare a logical winner.",
        judge_temp,
    )

    print(f"\n--- DEBATE TOPIC: {topic} ---\n")

    # Initialize conversation history
    conversation_history = []
    initial_prompt = f"The topic is: {topic}. Start the debate."
    conversation_history.append(HumanMessage(content=initial_prompt))

    # Track total tokens used
    total_tokens = 0

    # Run debate rounds dynamically
    round_count = 0
    while round_count < max_rounds:
        round_count += 1
        print(f"\n--- ROUND {round_count} ---")

        # Proponent speaks
        pro_input = "Present your argument."
        pro_response, pro_tokens = pro_agent.respond(pro_input, conversation_history)
        total_tokens += pro_tokens
        print(f"PRO (Tokens: {pro_tokens}): {pro_response}\n")
        conversation_history.append(AIMessage(content=f"Proponent: {pro_response}"))

        # Opponent responds
        con_input = "Respond to the proponent's argument."
        con_response, con_tokens = con_agent.respond(con_input, conversation_history)
        total_tokens += con_tokens
        print(f"CON (Tokens: {con_tokens}): {con_response}\n")
        conversation_history.append(AIMessage(content=f"Opponent: {con_response}"))

        # Ask the judge if we should continue
        if round_count < max_rounds:
            print("--- CHECKING IF DEBATE SHOULD CONTINUE ---")
            if not should_continue_debate(
                judge_agent, conversation_history, max_rounds
            ):
                print("Judge indicates sufficient information has been presented.\n")
                break
            else:
                print("Judge indicates more debate is needed.\n")

    # Judge's final verdict
    print(f"--- FINAL JUDGMENT (after {round_count} rounds) ---")
    judge_input = "Provide your final judgment based on all arguments presented. Summarize the key points of both sides and declare a logical winner."
    judge_response, judge_tokens = final_judge_agent.respond(
        judge_input, conversation_history
    )
    total_tokens += judge_tokens
    print(f"--- JUDGE'S VERDICT (Tokens: {judge_tokens}) ---\n{judge_response}")

    # Display total token usage
    print(f"\n--- TOKEN USAGE REPORT ---")
    print(f"Total tokens used: {total_tokens}")
    print(f"Total rounds: {round_count}")


if __name__ == "__main__":
    topic = (
        # "Should Linux distributions move away from X11 entirely in favor of Wayland?"
        "When parking your car in a parking lot, is it better to park straight-in or back-in?"
        # "The Standard Model of particle physics, which includes quarks, leptons, bosons, and antimatter, is a complete framework for understanding the fundamental nature of the Universe."
        # "String theory should be considered a failure since it has not been able to unify theories of everything by now. Scientists should stop spending resources on it and move on."
    )
    # Example with different temperatures for each persona
    run_debate(topic, pro_temp=0.9, con_temp=0.7, judge_temp=0.3, max_rounds=3)

import ollama


def chat_with_agent(model, system_prompt, conversation_history, temperature=0.8):
    # Combine system prompt with history
    messages = [{"role": "system", "content": system_prompt}] + conversation_history

    response = ollama.chat(
        model=model, messages=messages, options={"temperature": temperature}
    )

    # Extract token usage information if available
    tokens_used = 0
    if "prompt_eval_count" in response and "eval_count" in response:
        tokens_used = response["prompt_eval_count"] + response["eval_count"]
    elif "eval_count" in response:
        tokens_used = response["eval_count"]

    return response["message"]["content"], tokens_used


def run_debate(topic, rounds=2, pro_temp=0.8, con_temp=0.8, judge_temp=0.5):
    model_name = "my-gemma"

    # Personas
    PRO_PROMPT = "You are an optimistic advocate. Support the topic with logic and enthusiasm. Be brief."
    CON_PROMPT = "You are a skeptical critic. Find flaws in the opponent's logic and argue against the topic. Be brief."
    JUDGE_PROMPT = "You are a neutral judge. Summarize the key points of both sides and declare a logical winner."

    history = [{"role": "user", "content": f"The topic is: {topic}. Start the debate."}]

    # Track total tokens used
    total_tokens = 0

    print(f"\n--- DEBATE TOPIC: {topic} ---\n")

    for i in range(rounds):
        # Round i: Proponent speaks
        pro_speech, pro_tokens = chat_with_agent(
            model_name, PRO_PROMPT, history, pro_temp
        )
        total_tokens += pro_tokens
        print(f"PRO: {pro_speech}\n")
        history.append({"role": "assistant", "content": f"Proponent: {pro_speech}"})

        # Round i: Opponent responds
        con_speech, con_tokens = chat_with_agent(
            model_name, CON_PROMPT, history, con_temp
        )
        total_tokens += con_tokens
        print(f"CON: {con_speech}\n")
        history.append({"role": "assistant", "content": f"Opponent: {con_speech}"})

    # Final Judgment
    verdict, judge_tokens = chat_with_agent(
        model_name, JUDGE_PROMPT, history, judge_temp
    )
    total_tokens += judge_tokens
    print(f"--- JUDGE'S VERDICT ---\n{verdict}")

    # Report total tokens used
    print(f"\n--- TOKEN USAGE REPORT ---")
    print(f"Total tokens used: {total_tokens}")


if __name__ == "__main__":
    topic = (
        # "Should Linux distributions move away from X11 entirely in favor of Wayland?"
        "When parking your car in a parking lot, is it better to park straight-in or back-in?"
    )
    # Example with different temperatures for each persona
    run_debate(topic, pro_temp=0.9, con_temp=0.7, judge_temp=0.3)

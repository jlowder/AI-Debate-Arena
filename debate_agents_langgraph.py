import ollama
from typing import Annotated, List, TypedDict, Dict, Any
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
import operator


class DebateState(TypedDict):
    topic: str
    messages: Annotated[List[BaseMessage], operator.add]
    round_count: int
    max_rounds: int
    total_tokens: int
    pro_temp: float
    con_temp: float
    judge_temp: float
    should_continue: bool
    judge_reason: str
    final_verdict: str


def chat_with_ollama(model_name, messages, temperature):
    """Direct call to Ollama to get response and token information"""
    formatted_messages = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            formatted_messages.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            formatted_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            formatted_messages.append({"role": "assistant", "content": msg.content})

    response = ollama.chat(
        model=model_name,
        messages=formatted_messages,
        options={"temperature": temperature},
    )

    tokens_used = 0
    if "prompt_eval_count" in response and "eval_count" in response:
        tokens_used = response["prompt_eval_count"] + response["eval_count"]
    elif "eval_count" in response:
        tokens_used = response["eval_count"]

    return response["message"]["content"], tokens_used


def proponent_node(state: DebateState):
    model_name = "my-gemma"
    system_prompt = "You are an optimistic advocate. Support the topic with logic and enthusiasm. Be brief."

    current_messages = [SystemMessage(content=system_prompt)] + state["messages"]
    pro_input = "Present your argument."

    # We add the "human" instruction for the agent
    current_messages.append(HumanMessage(content=pro_input))

    response, tokens = chat_with_ollama(model_name, current_messages, state["pro_temp"])

    print(f"PRO (Tokens: {tokens}): {response}\n")

    return {
        "messages": [AIMessage(content=f"Proponent: {response}")],
        "total_tokens": state["total_tokens"] + tokens,
        "round_count": state["round_count"] + 1
    }


def opponent_node(state: DebateState):
    model_name = "my-gemma"
    system_prompt = "You are a skeptical critic. Find flaws in the opponent's logic and argue against the topic. Be brief."

    current_messages = [SystemMessage(content=system_prompt)] + state["messages"]
    con_input = "Respond to the proponent's argument."

    current_messages.append(HumanMessage(content=con_input))

    response, tokens = chat_with_ollama(model_name, current_messages, state["con_temp"])

    print(f"CON (Tokens: {tokens}): {response}\n")

    return {
        "messages": [AIMessage(content=f"Opponent: {response}")],
        "total_tokens": state["total_tokens"] + tokens
    }


def judge_node(state: DebateState):
    if state["round_count"] >= state["max_rounds"]:
        return {"should_continue": False, "judge_reason": "Maximum rounds reached."}

    model_name = "my-gemma"
    system_prompt = "You are a decisive judge who can make reasonable judgments on practical matters. You don't need perfect information to make a call - you can make a judgment based on the arguments presented even if they aren't exhaustive. When asked if the debate should continue, provide a brief explanation of your reasoning."

    judge_prompt = """
    Based on the debate so far, do you have enough information to make a final judgment, or should the debaters continue?

    You should recommend stopping the debate ("JUDGMENT READY") if:
    - Both sides have presented at least one argument
    - There has been some attempt at rebuttal
    - You can form a reasonable opinion based on what's been said, even if it's not definitive
    - The topic is one where a layperson can reasonably judge (like ethical or practical matters)

    You should recommend continuing ("CONTINUE") only if:
    - One side hasn't presented a meaningful argument
    - There's been no rebuttal from either side
    - Critical information is missing that would be necessary for any judgment

    Respond with either:
    "JUDGMENT READY" if you have enough information to make a final decision
    "CONTINUE" if more rounds would be beneficial

    Additionally, provide a brief explanation (1 sentence) of your reasoning after stating your decision.
    For example: "JUDGMENT READY - Both sides have presented arguments and there's enough information to make a reasonable judgment."
    Or: "CONTINUE - The con side hasn't provided a substantive counter-argument yet."

    Do not include any other text in your response.
    """

    current_messages = [SystemMessage(content=system_prompt)] + state["messages"]
    current_messages.append(HumanMessage(content=judge_prompt))

    response, tokens = chat_with_ollama(model_name, current_messages, state["judge_temp"])

    print(f"Judge's decision: {response}")

    # Parse the judge's response
    response_upper = response.strip().upper()
    # We should continue unless "JUDGMENT READY" is present, which is how it's done in lc version
    should_continue = "JUDGMENT READY" not in response_upper

    return {
        "should_continue": should_continue,
        "judge_reason": response,
        "total_tokens": state["total_tokens"] + tokens
    }


def final_judge_node(state: DebateState):
    model_name = "my-gemma"
    system_prompt = "You are a neutral judge. Summarize the key points of both sides and declare a logical winner."

    judge_input = "Provide your final judgment based on all arguments presented. Summarize the key points of both sides and declare a logical winner."

    current_messages = [SystemMessage(content=system_prompt)] + state["messages"]
    current_messages.append(HumanMessage(content=judge_input))

    response, tokens = chat_with_ollama(model_name, current_messages, state["judge_temp"])

    print(f"--- FINAL JUDGMENT (after {state['round_count']} rounds) ---")
    print(f"--- JUDGE'S VERDICT (Tokens: {tokens}) ---\n{response}")

    return {
        "final_verdict": response,
        "total_tokens": state["total_tokens"] + tokens
    }


def router(state: DebateState):
    if state["should_continue"]:
        return "proponent"
    else:
        return "final_judge"


def run_debate(topic, pro_temp=0.8, con_temp=0.8, judge_temp=0.5, max_rounds=10):
    # Initialize the graph
    workflow = StateGraph(DebateState)

    # Add nodes
    workflow.add_node("proponent", proponent_node)
    workflow.add_node("opponent", opponent_node)
    workflow.add_node("judge", judge_node)
    workflow.add_node("final_judge", final_judge_node)

    # Add edges
    workflow.add_edge(START, "proponent")
    workflow.add_edge("proponent", "opponent")
    workflow.add_edge("opponent", "judge")

    # Add conditional edges from judge
    workflow.add_conditional_edges(
        "judge",
        router,
        {
            "proponent": "proponent",
            "final_judge": "final_judge"
        }
    )

    workflow.add_edge("final_judge", END)

    # Compile the graph
    app = workflow.compile()

    print(f"\n--- DEBATE TOPIC: {topic} ---\n")

    initial_prompt = f"The topic is: {topic}. Start the debate."

    # Initial state
    initial_state = {
        "topic": topic,
        "messages": [HumanMessage(content=initial_prompt)],
        "round_count": 0,
        "max_rounds": max_rounds,
        "total_tokens": 0,
        "pro_temp": pro_temp,
        "con_temp": con_temp,
        "judge_temp": judge_temp,
        "should_continue": True,
        "judge_reason": "",
        "final_verdict": ""
    }

    # Run the graph
    final_state = app.invoke(initial_state)

    # Display total token usage
    print(f"\n--- TOKEN USAGE REPORT ---")
    print(f"Total tokens used: {final_state['total_tokens']}")
    print(f"Total rounds: {final_state['round_count']}")

    return final_state


if __name__ == "__main__":
    topic = "When parking your car in a parking lot, backing-in is better than pulling straight-in"
    run_debate(topic, pro_temp=0.9, con_temp=0.7, judge_temp=0.3, max_rounds=5)

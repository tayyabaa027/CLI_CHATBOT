"""
Multi-provider CLI chatbot with sliding-window memory and live cost tracking.
Run with: python -m chatbot.cli

New features:
  /clear        Wipe conversation history and start fresh
  /summary      Ask Gemini to summarize the conversation in 3 bullet points
  --persona     Launch with a custom system prompt  e.g. --persona "You are a Python tutor"
  Colors        Bot replies in green, your messages in white (via colorama)
  Token warning 80% token-limit warning before memory trimming kicks in
"""
import argparse
from dotenv import load_dotenv
load_dotenv()

from colorama import init as colorama_init, Fore, Style
colorama_init(autoreset=True)

from chatbot.memory import ConversationManager
from chatbot.providers import call_model
from chatbot.pricing import compute_cost

DEFAULT_MODELS = {
    "gemini": "gemini-2.0-flash-lite",
}

HELP_TEXT = f"""
{Fore.CYAN}Commands:
  /cost                Show running token usage and total spend
  /clear               Wipe conversation history and start fresh
  /summary             Summarize the conversation so far in 3 bullet points
  /save <file>         Save conversation history to a JSON file (default: conversation.json)
  /load <file>         Load conversation history from a JSON file
  /quit                Exit{Style.RESET_ALL}
"""

TOKEN_WARN_THRESHOLD = 0.80   # warn at 80 % of max_tokens


def _check_token_warning(memory: ConversationManager, print_fn) -> None:
    """Print a warning if we're at or above 80 % of the token budget."""
    used = memory.total_tokens()
    ratio = used / memory.max_tokens if memory.max_tokens else 0
    if ratio >= TOKEN_WARN_THRESHOLD:
        pct = int(ratio * 100)
        print_fn(
            f"{Fore.YELLOW}⚠  Token warning: {used}/{memory.max_tokens} tokens used "
            f"({pct}%). Oldest messages will be trimmed soon.{Style.RESET_ALL}"
        )


def run(provider: str, system_prompt: str, max_tokens: int, input_fn=input, print_fn=print):
    """
    Core REPL loop.  input_fn/print_fn are injectable so tests can run
    without touching real stdin/stdout.
    """
    model = DEFAULT_MODELS[provider]
    memory = ConversationManager(system_prompt=system_prompt, max_tokens=max_tokens)
    total_cost = 0.0
    total_tokens_used = 0

    print_fn(
        f"{Fore.CYAN}Chatting with {provider} ({model}). "
        f"Type /help for commands.{Style.RESET_ALL}"
    )

    while True:
        try:
            raw = input_fn(f"{Fore.WHITE}you> {Style.RESET_ALL}").strip()
        except (EOFError, KeyboardInterrupt):
            print_fn(f"\n{Fore.CYAN}bye.{Style.RESET_ALL}")
            return

        if not raw:
            continue

        # ── built-in commands ──────────────────────────────────────────────
        if raw == "/quit":
            return

        if raw == "/help":
            print_fn(HELP_TEXT)
            continue

        if raw == "/cost":
            print_fn(
                f"{Fore.CYAN}Total tokens used: {total_tokens_used} | "
                f"Total cost: ${total_cost:.6f}{Style.RESET_ALL}"
            )
            continue

        if raw == "/clear":
            memory.clear(system_prompt)
            total_cost = 0.0
            total_tokens_used = 0
            print_fn(f"{Fore.CYAN}Conversation cleared. Starting fresh.{Style.RESET_ALL}")
            continue

        if raw == "/summary":
            non_system = [m for m in memory.messages if m["role"] != "system"]
            if not non_system:
                print_fn(f"{Fore.YELLOW}Nothing to summarize yet.{Style.RESET_ALL}")
                continue
            # Build a one-shot request so the summary doesn't pollute history
            from chatbot.memory import ConversationManager as CM
            summary_mem = CM(
                system_prompt="You are a concise summarizer.",
                max_tokens=max_tokens,
            )
            convo_text = "\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in non_system
            )
            summary_mem.add(
                "user",
                f"Summarize this conversation in exactly 3 bullet points:\n\n{convo_text}",
            )
            reply, usage = call_model(provider, model, summary_mem.messages)
            cost = compute_cost(model, usage)
            total_cost += cost
            total_tokens_used += usage["input_tokens"] + usage["output_tokens"]
            print_fn(f"\n{Fore.GREEN}📋 Summary:\n{reply}{Style.RESET_ALL}\n")
            continue

        if raw.startswith("/save"):
            parts = raw.split(maxsplit=1)
            path = parts[1] if len(parts) == 2 else "conversation.json"
            memory.save(path)
            print_fn(f"{Fore.CYAN}Saved to {path}{Style.RESET_ALL}")
            continue

        if raw.startswith("/load"):
            parts = raw.split(maxsplit=1)
            path = parts[1] if len(parts) == 2 else "conversation.json"
            memory.load(path)
            print_fn(f"{Fore.CYAN}Loaded from {path}{Style.RESET_ALL}")
            continue

        # ── normal message ─────────────────────────────────────────────────
        memory.add("user", raw)

        # Token warning BEFORE the API call so the user sees it in time
        _check_token_warning(memory, print_fn)

        reply, usage = call_model(provider, model, memory.messages)
        memory.add("assistant", reply)

        cost = compute_cost(model, usage)
        total_cost += cost
        total_tokens_used += usage["input_tokens"] + usage["output_tokens"]

        print_fn(f"\n{Fore.GREEN}{provider}> {reply}{Style.RESET_ALL}")
        print_fn(
            f"   [{usage['input_tokens']}in / {usage['output_tokens']}out tokens | "
            f"${cost:.6f} this turn | ${total_cost:.6f} total]\n"
        )


def main():
    parser = argparse.ArgumentParser(description="Gemini CLI chatbot with memory")
    parser.add_argument("--provider", default="gemini", choices=DEFAULT_MODELS.keys())
    parser.add_argument(
        "--system",
        default="You are a helpful, concise assistant.",
        help="Base system prompt",
    )
    parser.add_argument(
        "--persona",
        default=None,
        help='Override the system prompt with a custom persona, e.g. --persona "You are a Python tutor"',
    )
    parser.add_argument(
        "--max-tokens", type=int, default=3000,
        help="Sliding-window token budget before old messages get dropped",
    )
    args = parser.parse_args()

    # --persona takes precedence over --system
    system_prompt = args.persona if args.persona else args.system
    run(args.provider, system_prompt, args.max_tokens)


if __name__ == "__main__":
    main()

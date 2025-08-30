from snake_ai_pytorch.models import SB3Agent

if __name__ == "__main__":
    # agent = SB3Agent()
    # agent = SB3Agent(render_mode="fast_training")
    agent = SB3Agent(render_mode="human")
    agent.train()

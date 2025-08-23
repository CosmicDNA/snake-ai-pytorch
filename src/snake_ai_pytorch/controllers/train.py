from snake_ai_pytorch.models import Agent

if __name__ == "__main__":
    # agent = Agent()
    # agent = Agent(render_mode="fast_training")
    agent = Agent(render_mode="human")
    agent.train()

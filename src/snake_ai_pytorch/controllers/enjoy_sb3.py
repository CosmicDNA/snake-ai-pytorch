from snake_ai_pytorch.models.sb3_agent import SB3Agent

if __name__ == "__main__":
    agent = SB3Agent(render_mode="human")
    agent.play()

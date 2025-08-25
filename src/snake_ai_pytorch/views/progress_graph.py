import multiprocessing as mp


class Plotting:
    def __init__(self, train):
        # We import pyplot in the child process to avoid issues with forking on some OSes.
        def plotter_process(queue, train):
            """Runs in a separate process to handle plotting."""
            import matplotlib.pyplot as plt

            plt.ion()
            fig, ax = plt.subplots()
            ax.set_title(f"{'Training' if train else 'Playing'}...")
            ax.set_xlabel("Number of Games")
            ax.set_ylabel("Score")
            (score_line,) = ax.plot([], [], label="Score")
            (mean_score_line,) = ax.plot([], [], label="Mean Score")
            score_text = ax.text(0, 0, "")
            mean_score_text = ax.text(0, 0, "")
            ax.legend(loc="upper left")

            scores = []
            mean_scores = []
            training_end_marker = None
            vline = None

            def update_plot_data():
                nonlocal vline

                x_data = range(len(scores))
                score_line.set_data(x_data, scores)
                mean_score_line.set_data(x_data, mean_scores)

                if scores:
                    last_game_idx = len(scores) - 1
                    score_text.set_position((last_game_idx, scores[-1]))
                    score_text.set_text(str(scores[-1]))
                    mean_score_text.set_position((last_game_idx, mean_scores[-1]))
                    mean_score_text.set_text(f"{mean_scores[-1]:.2f}")

                if training_end_marker is not None and vline is None:
                    vline = ax.axvline(x=training_end_marker, color="r", linestyle="--", label="End of Training")
                    ax.legend(loc="upper left")

                ax.relim()
                ax.autoscale_view()
                ax.set_ylim(bottom=0, top=max(scores))
                fig.canvas.draw()
                fig.canvas.flush_events()

            while True:
                try:
                    if not queue.empty():
                        data = queue.get()
                        if data is None:  # Sentinel for stopping
                            break

                        command, values = data
                        if command == "load":
                            scores, mean_scores = values
                        elif command == "plot":
                            score, mean_score = values
                            scores.append(score)
                            mean_scores.append(mean_score)
                        elif command == "add_marker":
                            training_end_marker = values
                        update_plot_data()

                    plt.pause(0.1)
                except (KeyboardInterrupt, BrokenPipeError):
                    break
                except Exception:  # Catches exceptions if the window is closed manually
                    break

            plt.ioff()
            plt.close(fig)

        # Use a multiprocessing queue for safe data exchange
        self.queue = mp.Queue()
        self.process = mp.Process(target=plotter_process, args=(self.queue, train), daemon=True)
        # The agent still needs to track scores for saving checkpoints
        self.scores = []
        self.mean_scores = []

    def start(self):
        self.process.start()

    def stop(self):
        """Send a signal to stop the plotting process."""
        if self.process.is_alive():
            self.queue.put(None)  # Send sentinel
            self.process.join(timeout=1)
            if self.process.is_alive():
                self.process.terminate()  # Forcefully stop if it doesn't close

    def load_data(self, scores, mean_scores):
        """Load existing data and send it to the plotting process."""
        self.scores = scores
        self.mean_scores = mean_scores
        self.queue.put(("load", (scores, mean_scores)))

    def plot(self, score, mean_score):
        """Append new data points and send them to the plotting process."""
        self.scores.append(score)
        self.mean_scores.append(mean_score)
        self.queue.put(("plot", (score, mean_score)))

    def add_training_marker(self, game_number):
        """Sends a command to draw the training/playing delimiter."""
        self.queue.put(("add_marker", game_number))

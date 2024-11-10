class TqdmToLogger:
    """Redirect tqdm output to logger with live updates"""
    def __init__(self, buf):
        self.buf = buf

    def write(self, x):
        if len(x.rstrip()) > 0:
            # Remove previous line if it exists
            if '\r' in x:
                # Clear the last line and move cursor up
                self.buf.write('\033[K')
            self.buf.write(x)
            self.buf.flush()  # Ensure immediate flush

    def flush(self):
        self.buf.flush() 
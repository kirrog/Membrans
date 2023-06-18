from datetime import datetime


class CstmLogger:
    def __init__(self, fptr):
        self.fptr = fptr

    def log(self, text: str):
        log_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_text = f"[{log_time}] - {text}"
        print(log_text)
        print(log_text, file=self.fptr)

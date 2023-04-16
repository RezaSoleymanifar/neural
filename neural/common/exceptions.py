class CorruptDataError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class IncompatibleWrapperError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class TradeConstraintViolationError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

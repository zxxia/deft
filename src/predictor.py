class SyncFreqPredictor:
    def __init__(self) -> None:
        pass
    def predict(self) -> int:
        return 50

# TODO: implement it
class ProfileBasedSyncFreqPredictor(SyncFreqPredictor):
    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError

    def predict(self) -> int:
        raise NotImplementedError
        # return super().predict()


# TODO: implement it
class MathModelBasedSyncFreqPredictor(SyncFreqPredictor):
    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError

    def predict(self) -> int:
        raise NotImplementedError
        # return super().predict()

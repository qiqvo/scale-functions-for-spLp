class StickBreakingRepresentationFactory():
    def __init__(self, sbr_type, **args) -> None:
        self.args = args
        self.sbr_type = sbr_type
        
    def create(self, process):
        self.sbr_type(process=process, **self.args)